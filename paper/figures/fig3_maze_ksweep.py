"""Figure 3: Maze accuracy and search cost against the number K of supervised shortest paths.

15x15 synthetic mazes; mean over seeds with the seed range. Solve rate for
argmax digits at evaluation (the default) and for sampled digits (the
`k_<K>_sample` entries, the same checkpoints), and model calls per solved
maze for argmax digits.

    uv run python -m paper.figures.fig3_maze_ksweep [--copy-to DIR]
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
    """{policy: {K: {"acc": (mean, min, max), "calls": mean calls per solved maze}}}."""
    out: dict = {"argmax": {}, "sample": {}}
    for all_recs in by_entry(records).values():
        recs = done(all_recs)
        entry = all_recs[0]["entry"]
        print(f"{entry}: seeds {seeds_cell(all_recs)}")
        if not recs:
            continue
        policy = "sample" if entry.endswith("_sample") else "argmax"
        k = int(entry.split("_")[1])
        acc = np.array([r["accuracy"] for r in recs])
        out[policy][k] = {"acc": (acc.mean(), acc.min(), acc.max()),
                          "calls": np.mean([r["calls_correct_mean"] for r in recs])}
    return {p: dict(sorted(c.items())) for p, c in out.items() if c}


def plot(curves: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for policy, style in (("argmax", "-"), ("sample", "--")):
        if policy not in curves:
            continue
        c = curves[policy]
        ks = list(c)
        mean, lo, hi = (np.array([100 * c[k]["acc"][i] for k in ks]) for i in range(3))
        ax.plot(ks, mean, style, marker="o", ms=3, color="C0",
                label=f"solve rate, {policy} digits")
        if policy == "argmax":               # seed range; the sampled range is much wider
            ax.fill_between(ks, lo, hi, color="C0", alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(curves["argmax"]), [str(k) for k in curves["argmax"]])
    ax.set_xlabel("Supervised shortest paths $K$")
    ax.set_ylabel("Solve rate (%)", color="C0")
    calls_ax = ax.twinx()
    c = curves["argmax"]
    calls_ax.plot(list(c), [c[k]["calls"] for k in c], "-s", ms=3, color="C2",
                  label="model calls / solved maze")
    calls_ax.set_ylabel("Model calls per solved maze", color="C2")
    calls_ax.set_ylim(0, None)
    lines = ax.get_legend_handles_labels()
    more = calls_ax.get_legend_handles_labels()
    ax.legend(lines[0] + more[0], lines[1] + more[1], frameon=False, fontsize=7,
              loc="lower right", handlelength=2.6)
    fig.tight_layout()
    return pdf_bytes(fig)


def main() -> None:
    args = cli("Figure 3 (Maze K sweep).", default_manifest="fig3_maze_ksweep")
    path = write_output("maze_k_sweep.pdf", plot(k_curve(read_results(args.manifest, args.results))),
                        args.out, args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
