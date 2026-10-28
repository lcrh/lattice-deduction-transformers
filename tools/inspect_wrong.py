"""Print the givens, prediction and solution of Sudoku puzzles a run got wrong or timed out on.

Rows come from a collected rows file (`paper.launch collect --rows`) or a
downloaded eval run directory; puzzles from the local Sudoku-Extreme cache.

    uv run python tools/inspect_wrong.py ROWS --cache data/sudoku_extreme_test.pt [--status wrong]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ldt.eval.records import read_blocks


def load_rows(path: Path) -> list[dict]:
    """Rows of a rows file or an eval run directory, in index-list order."""
    if path.is_dir():
        rows = [row for block in read_blocks(path).values() for row in block]
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["pos"])


def digits(lattice: torch.Tensor) -> str:
    """[81, 9] lattice -> 81 chars, '.' for cells that are not a singleton."""
    single = lattice.sum(-1) == 1
    vals = lattice.argmax(-1) + 1
    return "".join(str(int(v)) if s else "." for v, s in zip(vals, single))


def grid(cells: str) -> list[str]:
    cells = cells.replace("0", ".") if cells else "?" * 81
    lines = []
    for r in range(9):
        row = cells[9 * r:9 * r + 9]
        lines.append(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
        if r in (2, 5):
            lines.append("------+-------+------")
    return lines


def show(row: dict, x: torch.Tensor, y: torch.Tensor) -> str:
    givens, solution, pred = digits(x[row["idx"]]), digits(y[row["idx"]]), row.get("solution", "")
    head = (f"idx {row['idx']} (pos {row['pos']}): {row['status']}, rounds={row.get('rounds')}, "
            f"calls={row.get('calls')}, resets={row.get('n_resets')}")
    if pred and len(pred) == 81:
        diff = sum(p != s for p, s in zip(pred, solution))
        head += f", {diff} cells differ from the solution"
    cols = zip(grid(givens), grid(pred), grid(solution))
    body = "\n".join(f"{a:<22}   {b:<22}   {c}" for a, b, c in cols)
    return f"{head}\n{'givens':<22}   {'prediction':<22}   solution\n{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rows", type=Path, help="rows .jsonl or an eval run directory")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--status", choices=("wrong", "timeout", "both"), default="both")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    data = torch.load(args.cache, map_location="cpu", weights_only=True)
    wanted = {"wrong", "timeout"} if args.status == "both" else {args.status}
    bad = [r for r in load_rows(args.rows) if r["status"] in wanted]
    print(f"{len(bad)} rows with status in {sorted(wanted)}\n")
    for row in bad[:args.limit]:
        print(show(row, data["x"], data["y"]))


if __name__ == "__main__":
    main()
