"""Table 1: Sudoku-Extreme. LDT rows from table1_sudoku, baseline rows from
the baselines manifest's result files, and the portfolio row from
paper/out/portfolio.json (paper.portfolio analyze) when present.

    uv run python -m paper.tables.table1_sudoku [--copy-to DIR]
"""

from __future__ import annotations

import json
from pathlib import Path

from paper.manifest import REPO
from paper.tables.common import (
    OUT, Table, by_entry, cli, done, escape, eval_rule, mean_std, num, pct, pm, read_results,
    write_output,
)


def baseline_rows(records: list[dict], group: str, root: Path = REPO) -> list[dict]:
    """Result JSON of done baseline command runs in `group`."""
    out = []
    for rec in records:
        if rec.get("group") == group and rec.get("status") == "done" and rec.get("result_path"):
            out.append({"label": rec["label"],
                        "result": json.loads((root / rec["result_path"]).read_text())})
    return out


def build(records: list[dict], baselines: list[dict], portfolio: dict | None) -> Table:
    columns = ["Method", "$n$ / seed", "Acc. (\\%)", "Wrong", "Timeouts", "Calls/solve",
               "GPU s / puzzle", "Train (min)"]
    body = []
    for b in baselines:
        r = b["result"]
        lo, hi = r.get("exact_accuracy_ci", [None, None])
        body.append([escape(b["label"]), num(r["n"]),
                     f"{pct(r['exact_accuracy'])} [{pct(lo)}, {pct(hi)}]", num(r["wrong"]),
                     "--", "--", num(r.get("gpu_seconds", 0) / r["n"], 4), "--"])
    ldt = done(records)
    for name, recs in by_entry(ldt).items():
        ns = sorted({r["n"] for r in recs})
        body.append([
            escape(recs[0]["label"]),
            f"{ns[0]:,}" if len(ns) == 1 else f"{ns[0]:,}--{ns[-1]:,}",
            pm(*mean_std([r["accuracy"] for r in recs]), digits=3),
            num(sum(r["wrong"] for r in recs)), num(sum(r["timeout"] for r in recs)),
            num(mean_std([r["calls_mean"] for r in recs])[0]),
            num(mean_std([r["gpu_seconds"] / r["n"] for r in recs])[0], 4),
            num(mean_std([r["train_seconds"] / 60 if r.get("train_seconds") else None
                          for r in recs])[0]),
        ])
    if portfolio and portfolio.get("smallest_k") is not None:
        body.append([f"LDT portfolio, {portfolio['smallest_k']} models", num(portfolio["n_total"]),
                     "100.0", "0", "0", "--", "--", "--"])
    caption = ("Sudoku-Extreme test accuracy. LDT rows: mean $\\pm$ std over 3 seeds. "
               f"{eval_rule(ldt)}. Baselines: released checkpoints, exact match on the first "
               "10,000 puzzles of the same ordering, with a 95\\% Clopper--Pearson interval.")
    return Table(columns=columns, rows=body, caption=caption, label="tab:sudoku")


def main() -> None:
    args = cli("Table 1 (Sudoku-Extreme).", default_manifest="table1_sudoku")
    records = read_results(args.manifest, args.results)
    try:
        baselines = baseline_rows(read_results("baselines", args.results), "sudoku")
    except FileNotFoundError:
        baselines = []
    portfolio_path = OUT / "portfolio.json"
    portfolio = json.loads(portfolio_path.read_text()) if portfolio_path.exists() else None
    table = build(records, baselines, portfolio)
    print(f"wrote {write_output('table1_sudoku.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
