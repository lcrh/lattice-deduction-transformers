"""Table 1: Sudoku-Extreme accuracy and wrong answers (%).

Rows, in the paper's order:
  baselines   released checkpoints on the first 10,000 puzzles of
              `sudoku_test_perm` (manifest `baselines`); EqR as reported
  LDT rows    manifest `table1_sudoku`; accuracy and wrong answers pooled over
              seeds (the rows trained without per-step augmentation)
  10,000 search rounds
              the 4K-step models (seeds 0-2, manifest `portfolio`, which shares
              their checkpoints) with every 1,000-round timeout rerun at
              10,000 rounds; mean over seeds
  portfolios  manifest `portfolio` scored by paper.portfolio: three disjoint
              5-model portfolios under agree-2 (mean), and all 15 models under
              agree-3

    uv run python -m paper.tables.table1_sudoku [--copy-to DIR]
"""

from __future__ import annotations

import numpy as np

from paper.portfolio import Evals, _resolve, test_evals
from paper.tables.common import (
    Table, baseline_rows, by_entry, cli, done, escape, read_results, write_output,
)

BASELINES = ("HRM", "TRM-MLP", "Sotaku v2")
REPORTED = [("EqR (as reported)", 99.8, None)]
LDT_ROWS = [("LDT, 1K train steps", "steps1k_train_noaug"),
            ("LDT, 2K train steps", "steps2k_train_noaug"),
            ("LDT, 4K train steps", "steps4k_train_noaug")]
NO_AUG = ("\\quad no training augmentation", "noaug4k_evalaug")


def acc_cells(acc: float, wrong: float | None, digits: int = 3) -> list[str]:
    """Accuracy and wrong-answer rate, both in %."""
    return [f"{acc:.{digits}f}", "--" if wrong is None else f"{wrong:.5f}"]


def pooled(recs: list[dict]) -> tuple[float, float]:
    n = sum(r["n"] for r in recs)
    return 100 * sum(r["correct"] for r in recs) / n, 100 * sum(r["wrong"] for r in recs) / n


def portfolio_cells(ev: Evals, n: int, portfolios: list[list[int]], rule: str) -> list[str]:
    """Mean accuracy and wrong-answer rate of `portfolios` (rows of `ev`) under `rule`."""
    acc, wrong = [], []
    for rows in portfolios:
        chosen = _resolve(ev.answer[rows], ev.calls[rows], rule)[0] if len(rows) > 1 else ev.answer[rows[0]]
        acc.append(100 * (1 - (chosen != 0).sum() / n))
        wrong.append(100 * (chosen > 0).sum() / n)
    return acc_cells(float(np.mean(acc)), float(np.mean(wrong)), 5)


def build(records: list[dict], baselines: list[dict],
          portfolio: tuple[Evals, int] | None = None) -> Table:
    body = []
    for b in baselines:
        r = b["result"]
        if any(b["label"].startswith(name) for name in BASELINES):
            body.append([escape(b["label"]), *acc_cells(100 * r["exact_accuracy"], 100 * r["wrong"] / r["n"], 2)])
    body += [[label, *acc_cells(acc, wrong, 1)] for label, acc, wrong in REPORTED]
    entries = by_entry(records)
    for label, entry in LDT_ROWS:
        recs = done(entries.get(entry, []))
        if recs:
            body.append([label, *acc_cells(*pooled(recs))])
    if portfolio is not None:
        ev, n = portfolio
        body.append(["\\quad 10,000 search rounds", *portfolio_cells(ev, n, [[0], [1], [2]], "first")])
        body.append(["\\quad portfolio of 5 models, agree-2",
                     *portfolio_cells(ev, n, [list(range(0, 5)), list(range(5, 10)), list(range(10, 15))], "agree2")])
        body.append(["\\quad portfolio of 15 models, agree-3",
                     *portfolio_cells(ev, n, [list(range(15))], "agree3")])
    recs = done(entries.get(NO_AUG[1], []))
    if recs:
        body.append([NO_AUG[0], *acc_cells(*pooled(recs), 2)])
    caption = ("Sudoku-Extreme test accuracy and wrong answers (\\%). Baselines: released "
               "checkpoints on the first 10,000 puzzles of the test ordering. LDT: pooled over 3 "
               "seeds; portfolios use up to 10,000 rounds per model.")
    return Table(columns=["Model", "Acc. \\%", "Wrong \\%"], rows=body, caption=caption,
                 label="tab:Sudoku-extreme")


def main() -> None:
    args = cli("Table 1 (Sudoku-Extreme).", default_manifest="table1_sudoku")
    records = read_results(args.manifest, args.results)
    table = build(records, baseline_rows("sudoku", args.results),
                  test_evals(read_results("portfolio", args.results)))
    print(f"wrote {write_output('table1_sudoku.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
