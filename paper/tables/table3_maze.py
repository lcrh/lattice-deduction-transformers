"""Maze-Hard table: lenient (any shortest path) and exact (reference path) accuracy.

LDT rows report each seed's best checkpoint, over the step snapshots and the
final checkpoint (manifest `table3_maze`). Baselines come from the HRM/TRM
checkpoint sweeps at the baselines manifest's `maze_step`: `best` (highest
lenient accuracy, chosen on the test set), `last`, or a step number.

    uv run python -m paper.tables.table3_maze [--copy-to DIR]
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from paper.manifest import REPO, table_hints
from paper.tables.common import (
    Table, baseline_rows, cli, done, escape, load_rows, mean_std, num, pct, pm,
    read_results, write_output,
)


MazeStep = int | Literal["best", "last"]


def _sweep_point(sweep: list[dict], step: MazeStep) -> dict:
    points = sorted(sweep, key=lambda p: p["step"])
    if step == "last":
        return points[-1]
    if step == "best":
        return max(points, key=lambda p: p["lenient"])
    return next(p for p in points if p["step"] == int(step))


def _best_checkpoints(records: list[dict], train_entry: str) -> list[dict]:
    """Per seed, the most accurate eval: a step snapshot or the final checkpoint."""
    cands = [r for r in done(records)
             if r["entry"] == train_entry or r["entry"].startswith(f"{train_entry}_snap_step")]
    best = {}
    for r in cands:
        if r["seed"] not in best or r["accuracy"] > best[r["seed"]]["accuracy"]:
            best[r["seed"]] = r
    return [best[s] for s in sorted(best)]


def build(records: list[dict], baselines: list[dict], maze_step: MazeStep = "last",
          root: Path = REPO) -> Table:
    columns = ["Method", "Seeds", "Lenient acc. (\\%)", "Exact acc. (\\%)", "Wrong", "Timeouts",
               "Steps"]
    body = []
    for b in baselines:
        p = _sweep_point(b["result"], maze_step)
        # The baselines always answer: every miss is a wrong answer.
        body.append([escape(b["label"]), "1", pct(p["lenient"] / p["n"]), pct(p["exact"] / p["n"]),
                     num(p["n"] - p["lenient"]), "0", f"{p['step']:,}"])
    for train_entry, label in (("k1", "LDT, K=1"), ("k512", "LDT, K=512")):
        recs = _best_checkpoints(records, train_entry)
        if not recs:
            continue
        exact = []
        for rec in recs:
            rows = load_rows(rec, root)
            exact.append(sum(r.get("label") == "CORRECT_GT" for r in rows) / len(rows))
        steps = [int(r["entry"].rsplit("step", 1)[1]) if "_snap_step" in r["entry"] else r["train_steps"]
                 for r in recs]
        body.append([label, str(len(recs)), pm(*mean_std([r["accuracy"] for r in recs])),
                     pm(*mean_std(exact)), num(sum(r["wrong"] for r in recs)),
                     num(sum(r["timeout"] for r in recs)), ", ".join(f"{int(s):,}" for s in steps)])
    selection = {"best": "best checkpoint by lenient test accuracy",
                 "last": "final checkpoint"}.get(str(maze_step), f"step {maze_step}")
    caption = ("Maze-Hard (30$\\times$30) test accuracy on all 1,000 test mazes. Lenient: any "
               "shortest path; exact: the dataset's reference path. Wrong: an answer that is "
               "not a shortest path; timeouts: no answer within the round budget, summed over "
               "seeds. LDT: each seed's best checkpoint (step snapshots and the final checkpoint, "
               "$\\theta_{\\text{elim}} = 0.1$), mean $\\pm$ std over seeds. "
               f"Baselines: {selection}. Steps: the reported checkpoint of each run.")
    return Table(columns=columns, rows=body, caption=caption, label="tab:maze-hard")


def main() -> None:
    args = cli("Table 3 (Maze-Hard).", default_manifest="table3_maze")
    table = build(read_results(args.manifest, args.results), baseline_rows("maze", args.results),
                  table_hints("baselines").get("maze_step", "last"))
    print(f"wrote {write_output('table3_maze.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
