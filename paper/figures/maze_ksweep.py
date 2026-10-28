"""Figure 6(b): Maze accuracy and search cost against the number K of supervised shortest paths.

15x15 synthetic mazes; solve rate (mean over seeds with the seed range) and
model calls per solved maze.

    uv run python -m paper.figures.maze_ksweep [--copy-to DIR]
"""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from paper.tables.common import (  # noqa: E402
    by_entry, cli, done, pdf_bytes, read_results, seeds_cell, write_output,
)


def k_curve(records: list[dict]) -> dict:
    """{K: {"acc": (mean, min, max), "calls": (mean, min, max) model calls per solved maze}}."""
    out: dict = {}
    for all_recs in by_entry(records).values():
        recs = done(all_recs)
        entry = all_recs[0]["entry"]
        print(f"{entry}: seeds {seeds_cell(all_recs)}")
        if not recs:
            continue
        k = int(entry.split("_")[1])
        acc = np.array([r["accuracy"] for r in recs])
        calls = np.array([r["calls_correct_mean"] for r in recs])
        out[k] = {"acc": (acc.mean(), acc.min(), acc.max()),
                  "calls": (calls.mean(), calls.min(), calls.max())}
    return dict(sorted(out.items()))


def plot(c: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(4.2, 3))
    calls_ax = ax.twinx()
    ks = list(c)
    acc = [np.array([100 * c[k]["acc"][i] for k in ks]) for i in range(3)]
    calls = [np.array([c[k]["calls"][i] for k in ks]) for i in range(3)]
    for axis, (mean, lo, hi), style, color, label in (
            (ax, acc, "-o", "C0", "solve rate"),
            (calls_ax, calls, "-s", "C2", "model calls / solved maze")):
        axis.plot(ks, mean, style, ms=4, lw=2, color=color, label=label)
        axis.fill_between(ks, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Supervised shortest paths $K$")
    ax.set_ylabel("Solve rate (%)", color="C0")
    ax.tick_params(axis="y", colors="C0")
    calls_ax.set_ylabel("Model calls per solved maze", color="C2")
    calls_ax.tick_params(axis="y", colors="C2")
    # Scale the calls axis from 0 so both curves start at the same height at K = 1.
    lo, hi = ax.get_ylim()
    frac = (acc[0][0] - lo) / (hi - lo)
    calls_ax.set_ylim(0, max(calls[0][0] / frac, 1.05 * calls[2].max()))
    lines = ax.get_legend_handles_labels()
    more = calls_ax.get_legend_handles_labels()
    ax.legend(lines[0] + more[0], lines[1] + more[1], fontsize=8, loc="center right")
    fig.tight_layout()
    return pdf_bytes(fig)


def main() -> None:
    args = cli("Figure 6(b) (Maze K sweep).", default_manifest="maze_ksweep")
    path = write_output("maze_k_sweep.pdf", plot(k_curve(read_results(args.manifest, args.results))),
                        args.out, args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
