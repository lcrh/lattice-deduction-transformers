"""Figure 3: Maze accuracy against the number K of supervised shortest paths.

15x15 synthetic mazes; mean over seeds with the seed range.

    uv run python -m paper.figures.fig3_maze_ksweep [--copy-to DIR]
"""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from paper.tables.common import by_entry, cli, done, pdf_bytes, read_results, write_output  # noqa: E402


def k_curve(records: list[dict]) -> dict:
    """{K: (mean, min, max)} of accuracy over seeds."""
    out = {}
    for recs in by_entry(done(records)).values():
        acc = np.array([r["accuracy"] for r in recs])
        out[int(recs[0]["x"])] = (acc.mean(), acc.min(), acc.max())
    return dict(sorted(out.items()))


def plot(curve: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 3))
    ks = list(curve)
    mean = [100 * curve[k][0] for k in ks]
    ax.errorbar(ks, mean, yerr=[[100 * (curve[k][0] - curve[k][1]) for k in ks],
                                [100 * (curve[k][2] - curve[k][0]) for k in ks]],
                marker="o", ms=3, capsize=2, color="0.2")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Supervised shortest paths K")
    ax.set_ylabel("Test accuracy (%)")
    fig.tight_layout()
    return pdf_bytes(fig)


def main() -> None:
    args = cli("Figure 3 (Maze K sweep).", default_manifest="fig3_maze_ksweep")
    path = write_output("maze_k_sweep.pdf", plot(k_curve(read_results(args.manifest, args.results))),
                        args.out, args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
