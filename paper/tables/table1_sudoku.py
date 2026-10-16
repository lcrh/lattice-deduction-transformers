"""Table 1: Sudoku-Extreme.

LDT rows from table1_sudoku, baseline rows from the baselines manifest, and
the portfolio rows from paper/out/portfolio.json when present.

    uv run python -m paper.tables.table1_sudoku [--copy-to DIR]
"""

from __future__ import annotations

import json

from lattice_diffusion.eval.stats import binomial_upper

from paper.tables.common import (
    OUT, Table, baseline_rows, by_entry, cli, done, escape, eval_rule, mean, mean_std, n_range,
    num, pct, pm, read_results, seeds_cell, seeds_text, write_output,
)


def wrong_cell(wrong: int, n: int) -> str:
    """Wrong answers out of n with the one-sided 95% upper bound on the wrong-answer rate."""
    return f"{wrong:,}/{n:,} ($\\leq${pct(binomial_upper(wrong, n), 2)}\\%)"


def portfolio_rows(portfolio: dict) -> list[list[str]]:
    """Oracle (union of solved), first-answer and majority rows at the smallest zero-residue k."""
    k = portfolio["smallest_k"] or portfolio["n_models"]
    n = portfolio["n_total"]
    residue = portfolio["residue_natural_order"][k - 1]
    rows = [[f"LDT portfolio, {k} models, oracle", "--", num(n),
             pct((n - residue) / n, 3), "--", num(residue), "--", "--", "--"]]
    for rule, label in (("first", "first answer"), ("majority", "majority vote")):
        sel = portfolio["selection"][rule][k - 1]
        wrong, abstain = sel.get("wrong", 0), sel.get("unsolved", 0)
        rows.append([f"LDT portfolio, {k} models, {label}", "--", num(n),
                     pct((n - wrong - abstain) / n, 3), wrong_cell(wrong, n), num(abstain),
                     "--", "--", "--"])
    return rows


def build(records: list[dict], baselines: list[dict], portfolio: dict | None) -> Table:
    columns = ["Method", "Seeds", "$n$ / seed", "Acc. (\\%)", "Wrong", "Timeouts",
               "Calls/solve", "GPU s / puzzle", "Train (min)"]
    body = []
    for b in baselines:
        r = b["result"]
        lo, hi = r.get("exact_accuracy_ci", [None, None])
        body.append([escape(b["label"]), "--", num(r["n"]),
                     f"{pct(r['exact_accuracy'])} [{pct(lo)}, {pct(hi)}]",
                     wrong_cell(r["wrong"], r["n"]), "0", "--",
                     num(r.get("gpu_seconds", 0) / r["n"], 4), "--"])
    ldt = done(records)
    for all_recs in by_entry(records).values():
        recs = done(all_recs)
        seeds = seeds_cell(all_recs)
        if not recs:
            continue
        body.append([
            escape(recs[0]["label"]), seeds,
            n_range([r["n"] for r in recs]),
            pm(*mean_std([r["accuracy"] for r in recs]), digits=3),
            wrong_cell(sum(r["wrong"] for r in recs), sum(r["n"] for r in recs)),
            num(sum(r["timeout"] for r in recs)),
            num(mean([r["calls_per_solve"] for r in recs])),
            # GPU time covers every evaluated puzzle, including past the stopping point.
            num(mean([r["gpu_seconds"] / r["n_evaluated"] for r in recs]), 4),
            # Post-compile training loop, as in paper.compute.
            num(mean([r["train_post_compile_seconds"] / 60
                      if r.get("train_post_compile_seconds") else None
                      for r in recs])),
        ])
    if portfolio:
        body += portfolio_rows(portfolio)
    caption = (f"Sudoku-Extreme test accuracy. LDT rows: mean $\\pm$ std over "
               f"{seeds_text(records)} seeds; wrong answers pooled over seeds, with the "
               "one-sided 95\\% Clopper--Pearson upper bound on their rate. "
               f"{eval_rule(ldt)}. Baselines: released checkpoints, exact match on the first "
               "10,000 puzzles of the same ordering, with a 95\\% Clopper--Pearson interval. "
               "Portfolio: oracle counts a puzzle as solved if any member solved it; first "
               "answer and majority vote need no ground truth (abstentions under timeouts).")
    return Table(columns=columns, rows=body, caption=caption, label="tab:sudoku")


def main() -> None:
    args = cli("Table 1 (Sudoku-Extreme).", default_manifest="table1_sudoku")
    records = read_results(args.manifest, args.results)
    baselines = baseline_rows("sudoku", args.results)
    portfolio_path = OUT / "portfolio.json"
    portfolio = json.loads(portfolio_path.read_text()) if portfolio_path.exists() else None
    table = build(records, baselines, portfolio)
    print(f"wrote {write_output('table1_sudoku.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
