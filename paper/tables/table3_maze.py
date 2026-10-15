"""Table 3: Maze-Hard. LDT is accepted on any shortest path (lenient); the
exact column counts only the dataset's reference path. Baselines come from
the HRM/TRM checkpoint sweeps, at the step named by the baselines manifest
(`maze_step: best` = the checkpoint with the highest lenient accuracy,
selected on the test set; `last` = the trajectory's final checkpoint; or an
explicit step).

    uv run python -m paper.tables.table3_maze [--copy-to DIR]
"""

from __future__ import annotations

from pathlib import Path

from paper.manifest import REPO, load_manifest
from paper.tables.common import (
    Table, by_entry, cli, done, escape, load_rows, mean_std, num, pct, pm, read_results,
    write_output,
)
from paper.tables.table1_sudoku import baseline_rows


def _sweep_point(sweep: list[dict], step) -> dict:
    points = sorted(sweep, key=lambda p: p["step"])
    if step == "last":
        return points[-1]
    if step == "best":     # selected on the test set, by the lenient criterion
        return max(points, key=lambda p: p["lenient"])
    return next(p for p in points if p["step"] == int(step))


def build(records: list[dict], baselines: list[dict], maze_step="last",
          root: Path = REPO) -> Table:
    columns = ["Method", "Seeds", "Lenient acc. (\\%)", "Exact acc. (\\%)", "Wrong"]
    body = []
    for b in baselines:
        p = _sweep_point(b["result"], maze_step)
        body.append([escape(b["label"]) + f" (step {p['step']:,})", "1",
                     pct(p["lenient"] / p["n"]), pct(p["exact"] / p["n"]),
                     num(p["n"] - p["lenient"])])
    for name, recs in by_entry(done(records)).items():
        exact = []
        for rec in recs:
            rows = load_rows(rec, root)
            exact.append(sum(r.get("label") == "CORRECT_GT" for r in rows) / len(rows))
        body.append([escape(recs[0]["label"]), str(len(recs)),
                     pm(*mean_std([r["accuracy"] for r in recs])), pm(*mean_std(exact)),
                     num(sum(r["wrong"] for r in recs))])
    selection = {"best": "best checkpoint by lenient test accuracy",
                 "last": "final checkpoint"}.get(str(maze_step), f"step {maze_step}")
    caption = ("Maze-Hard (30$\\times$30) test accuracy on all 1,000 test mazes. Lenient: any "
               "shortest path; exact: the dataset's reference path. LDT: mean $\\pm$ std over "
               f"seeds. Baselines: {selection}.")
    return Table(columns=columns, rows=body, caption=caption, label="tab:maze")


def main() -> None:
    args = cli("Table 3 (Maze-Hard).", default_manifest="table3_maze")
    bm, _ = load_manifest("baselines")
    try:
        baselines = baseline_rows(read_results("baselines", args.results), "maze")
    except FileNotFoundError:
        baselines = []
    table = build(read_results(args.manifest, args.results), baselines,
                  (bm.get("table") or {}).get("maze_step", "last"))
    print(f"wrote {write_output('table3_maze.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
