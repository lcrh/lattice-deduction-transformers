"""Figure 4: Sudoku-Extreme test-time compute against training compute.

(a) Percentiles of sequential forward passes per solved puzzle (the
    drained-chain estimate `forwards_seq`: cost if one chain ran at a time)
    against training examples, averaged over seeds, with the solve rate.
(b) Density of the same quantity per log10 unit for selected training
    budgets (manifest hint `histogram_x`), pooled over seeds.
(c) Fractions of test puzzles solved and answered wrong within a round
    budget R, for the 15 same-data 4K-step models of the `portfolio`
    manifest: their 1,000-round test evals, with timeouts replaced by their
    10,000-round evals on the hard set. Mean over models, range shaded.

    uv run python -m paper.figures.sudoku_compute [--copy-to DIR]
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from paper.manifest import REPO, table_hints  # noqa: E402
from paper.tables.common import (  # noqa: E402
    by_entry, cli, done, load_rows, pdf_bytes, read_results, seeds_cell, write_output,
)

PERCENTILES = (50, 75, 90, 95)
MIN_SOLVE = 0.99        # panel (a) shows budgets that solve at least this fraction
ROUND_BUDGETS = np.unique(np.logspace(0, 4, 41).astype(int))   # panel (c), 1 to 10,000


def solved_forwards(record: dict, root: Path = REPO) -> np.ndarray:
    """Drained-chain forwards of each solved puzzle (needs solve.estimate_sequential)."""
    return np.array([r["forwards_seq"] for r in load_rows(record, root)
                     if r["status"] == "correct"])


def percentile_curves(records: list[dict], root: Path = REPO) -> dict:
    """{steps: {p: mean over seeds, "solve": mean solve rate, "examples": train examples}}."""
    out = {}
    for all_recs in by_entry(records).values():
        recs = done(all_recs)
        print(f"{all_recs[0]['entry']}: seeds {seeds_cell(all_recs)}")
        if not recs:
            continue
        steps = recs[0]["x"] if recs[0].get("x") is not None else recs[0]["train_steps"]
        per_seed = np.array([np.percentile(solved_forwards(r, root), PERCENTILES) for r in recs])
        out[steps] = {**{p: per_seed[:, i].mean() for i, p in enumerate(PERCENTILES)},
                      "solve": np.mean([r["accuracy"] for r in recs]),
                      "examples": steps * recs[0]["train_batch"]}
    return dict(sorted(out.items()))


def budget_counts(first: list[dict], retry: list[dict]) -> dict[str, Counter]:
    """{status: Counter(rounds)} for one model, its timeouts replaced by the retry eval."""
    by_idx = {r["idx"]: r for r in retry}
    counts: dict[str, Counter] = {"correct": Counter(), "wrong": Counter(), "timeout": Counter()}
    for r in first:
        if r["status"] == "timeout" and r["idx"] in by_idx:
            r = by_idx[r["idx"]]
        counts[r["status"]][r["rounds"]] += 1
    return counts


def budget_curves(models: list[dict[str, Counter]]) -> dict[str, np.ndarray]:
    """[model, budget] fractions of puzzles solved / answered wrong within each round budget."""
    out = {"solved": [], "wrong": []}
    for counts in models:
        n = sum(sum(c.values()) for c in counts.values())
        for key, status in (("solved", "correct"), ("wrong", "wrong")):
            rounds = np.array(sorted(counts[status].elements()))
            out[key].append(np.searchsorted(rounds, ROUND_BUDGETS, side="right") / n)
    return {k: np.array(v) for k, v in out.items()}


def plot(curves: dict, densities: dict[int, np.ndarray], budget: dict[str, np.ndarray]) -> bytes:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(10.5, 2.9))
    steps = [s for s in curves if curves[s]["solve"] >= MIN_SOLVE] or list(curves)
    x = [curves[s]["examples"] for s in steps]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(PERCENTILES)))
    ax_f = ax_a.twinx()
    for p, c in zip(PERCENTILES, colors):
        ax_f.plot(x, [curves[s][p] for s in steps], marker="o", ms=3, color=c, label=f"p{p}")
    ax_f.set_yscale("log")
    lo = min(curves[s][PERCENTILES[0]] for s in steps)
    hi = max(curves[s][PERCENTILES[-1]] for s in steps)
    ax_f.set_ylim(lo / 30, hi * 1.5)              # room for the legend below the curves
    ax_f.set_ylabel("Test compute (forwards)")
    ax_a.plot(x, [curves[s]["solve"] for s in steps], "k--", marker="s", ms=3,
              label="solve rate")
    ax_a.set_xscale("log")
    ax_a.set_xticks(x, [f"{e / 1e6:.2g}M" for e in x])
    ax_a.minorticks_off()
    ax_a.set_ylim(min(0.99, min(curves[s]["solve"] for s in steps) - 0.002), 1.0005)
    ax_a.set_xlabel("Train compute (examples)")
    ax_a.set_ylabel("Solve rate")
    lines = ax_a.get_legend_handles_labels()
    more = ax_f.get_legend_handles_labels()
    ax_a.legend(lines[0] + more[0], lines[1] + more[1], frameon=False, fontsize=7, ncol=2,
                loc="lower right")
    ax_a.set_title("(a)", loc="left", fontsize=9)

    bins = np.linspace(0, 5.5, 221)
    kernel = np.exp(-0.5 * (np.arange(-20, 21) / 7) ** 2)
    for (s, fwd), c in zip(densities.items(),
                           plt.cm.plasma(np.linspace(0.15, 0.75, len(densities)))):
        dens, edges = np.histogram(np.log10(fwd), bins=bins, density=True)
        dens = np.convolve(dens, kernel / kernel.sum(), mode="same")   # smoothing
        mid = (edges[:-1] + edges[1:]) / 2
        ax_b.plot(10 ** mid, dens, color=c, label=f"{s // 1000}k train steps")
        ax_b.fill_between(10 ** mid, dens, color=c, alpha=0.2)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Per-puzzle forwards (sequential)")
    ax_b.set_ylabel("Density (per log$_{10}$ unit)")
    ax_b.legend(frameon=False, fontsize=7)
    ax_b.set_title("(b)", loc="left", fontsize=9)

    for key, label, c in (("solved", "unsolved (abstains)", "C0"), ("wrong", "wrong", "C3")):
        frac = 1 - budget[key] if key == "solved" else budget[key]
        frac = np.maximum(frac, 1e-7)                # log axis: zero wrong answers at small R
        ax_c.plot(ROUND_BUDGETS, frac.mean(0), color=c, label=label)
        ax_c.fill_between(ROUND_BUDGETS, frac.min(0), frac.max(0), color=c, alpha=0.2)
    ax_c.axvline(1000, color="0.6", ls=":", lw=1)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_ylim(1e-6, 1)
    ax_c.set_xlabel("Round budget")
    ax_c.set_ylabel("Fraction of test puzzles")
    ax_c.legend(frameon=False, fontsize=7)
    ax_c.set_title("(c)", loc="left", fontsize=9)
    fig.tight_layout()
    return pdf_bytes(fig)


def portfolio_budget(results: Path) -> dict[str, np.ndarray]:
    """Panel (c) curves from the portfolio manifest's collected rows."""
    recs = done(read_results("portfolio", results))
    retry = {r["seed"]: load_rows(r) for r in recs if r["entry"] == "same_data_test_r10k"}
    return budget_curves([budget_counts(load_rows(r), retry.get(r["seed"], []))
                          for r in recs if r["entry"] == "same_data_test"])


def main() -> None:
    args = cli("Figure 4 (Sudoku compute).", default_manifest="sudoku_compute")
    records = read_results(args.manifest, args.results)
    curves = percentile_curves(records)
    hist_x = table_hints(args.manifest).get("histogram_x", [max(curves)])
    densities = {x: np.concatenate([solved_forwards(r) for r in done(records)
                                    if r.get("x") == x])
                 for x in hist_x}
    path = write_output("sudoku_train_test_compute.pdf",
                        plot(curves, densities, portfolio_budget(args.results)),
                        args.out, args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
