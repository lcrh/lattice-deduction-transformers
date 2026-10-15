"""Figure 2: Sudoku-Extreme test-time compute against training compute.

(a) Percentiles of model calls per solved puzzle (calls while the puzzle
    occupied a solver slot, i.e. sequential forward passes of its chains)
    against training steps; each percentile is computed per seed, then
    averaged, with the seed range shaded.
(b) Histogram of calls per solved puzzle for the 4K-step model, pooled over
    seeds (log-spaced bins); the deduction-only and search modes separate.

    uv run python -m paper.figures.fig2_sudoku_compute [--copy-to DIR]
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from paper.manifest import REPO, load_manifest  # noqa: E402
from paper.tables.common import (  # noqa: E402
    by_entry, cli, done, load_rows, read_results, write_output,
)

PERCENTILES = (50, 75, 90, 95)


def solved_calls(record: dict, root: Path = REPO) -> np.ndarray:
    return np.array([r["calls"] for r in load_rows(record, root) if r["status"] == "correct"])


def percentile_curves(records: list[dict], root: Path = REPO) -> dict:
    """{steps: {p: (mean, lo, hi)}} over seeds."""
    out = {}
    for _, recs in by_entry(done(records)).items():
        steps = recs[0]["x"] if recs[0].get("x") is not None else recs[0]["train_steps"]
        per_seed = [np.percentile(solved_calls(r, root), PERCENTILES) for r in recs]
        arr = np.array(per_seed)
        out[steps] = {p: (arr[:, i].mean(), arr[:, i].min(), arr[:, i].max())
                      for i, p in enumerate(PERCENTILES)}
    return dict(sorted(out.items()))


def plot(curves: dict, hist_calls: np.ndarray) -> bytes:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.2))
    steps = list(curves)
    for p in PERCENTILES:
        mean = [curves[s][p][0] for s in steps]
        lo = [curves[s][p][1] for s in steps]
        hi = [curves[s][p][2] for s in steps]
        ax_a.plot(steps, mean, marker="o", ms=3, label=f"p{p}")
        ax_a.fill_between(steps, lo, hi, alpha=0.15)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Training steps")
    ax_a.set_ylabel("Model calls per solved puzzle")
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.set_title("(a)", loc="left", fontsize=9)

    if hist_calls.size:
        bins = np.logspace(0, np.log10(max(hist_calls.max(), 2)), 40)
        ax_b.hist(hist_calls, bins=bins, color="0.35")
        ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Model calls per solved puzzle")
    ax_b.set_ylabel("Puzzles")
    ax_b.set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def main() -> None:
    args = cli("Figure 2 (Sudoku compute).", default_manifest="fig2_sudoku_compute")
    m, _ = load_manifest(args.manifest)
    records = read_results(args.manifest, args.results)
    curves = percentile_curves(records)
    hist_x = (m.get("table") or {}).get("histogram_x", max(curves))
    hist = [r for r in done(records) if r.get("x") == hist_x]
    calls = np.concatenate([solved_calls(r) for r in hist]) if hist else np.array([])
    path = write_output("sudoku_train_test_compute.pdf", plot(curves, calls), args.out,
                        args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
