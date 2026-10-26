"""Figure 1(b): Sudoku-Extreme compute (training + solving the whole test set)
against accuracy on a nines scale, with the Pareto frontier.

Accuracies are Table 1's (paper.tables.table1_sudoku). Training PFLOPs and
inference TFLOPs per puzzle are the estimates of paper.compute
(`uv run python -m paper.compute table1_sudoku`); the portfolio's 2.04 models
per puzzle under agree-2 comes from the held-out simulation of paper.portfolio.

    uv run python -m paper.figures.fig1_frontier [--copy-to DIR]
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

from paper.tables.common import cli, pdf_bytes, write_output  # noqa: E402

N_TEST = 422_786
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.left": False})


def err(acc: float) -> float:
    """Error rate of an accuracy in %."""
    return 1 - acc / 100


# (label, training PFLOPs, inference TFLOPs per puzzle, accuracy %, is LDT, marker)
points = [("HRM (55.4%)", 1466, 0.22, 55.4, False, "s"),
          ("TRM (79.5%)", 2136, 0.41, 79.5, False, "s"),
          ("Sotaku v2", 1265, 0.14, 98.96, False, "s"),
          ("EqR", 1640, 210, 99.8, False, "s"),
          ("LDT, no train aug.", 18.7, 0.85, 97.68, True, "o"),
          ("LDT", 18.7, 1.5, 99.999, True, "o"),
          # agree-2 runs about 2.04 models per puzzle (held-out simulation); accuracy is
          # the mean of three 5-model portfolios (0, 2 and 3 failures).
          ("LDT, portfolio of 5", 5 * 18.7, 2.04 * 1.5, 100 * (1 - (5 / 3) / N_TEST), True, "^")]


def frontier(xs, ys):
    """Points no cheaper point beats on accuracy, sorted by cost, as a staircase."""
    best, out = None, []
    for x, y in sorted(zip(xs, ys)):
        if best is None or y < best:
            if out:
                out.append((x, out[-1][1]))
            out.append((x, y))
            best = y
    return zip(*out)


def plot() -> bytes:
    xlim = (100, 3e5)
    fig, ax = plt.subplots(figsize=(2.7, 2.2))
    xs = [tr + inf * N_TEST / 1000 for _, tr, inf, *_ in points]
    ys = [err(p[3]) for p in points]
    fx, fy = (list(v) for v in frontier(xs, ys))
    fx.append(xlim[1])                          # the best accuracy holds for any larger budget
    fy.append(fy[-1])
    ax.set_xlim(*xlim)
    ax.plot(fx, fy, color="0.55", lw=1, ls=(0, (4, 2)), zorder=1)
    offsets = [(6, -3), (6, -1.5), (6, -3), (6, -3), (0, -10), (-6, 2), (6, 2)]
    for (name, *_, ldt, marker), x, y, off in zip(points, xs, ys, offsets):
        color = "C3" if ldt else "0.4"
        ax.scatter(x, y, color=color, marker=marker, s=24 if marker != "^" else 32, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                    ha="left" if off[0] > 0 else "right" if off[0] < 0 else "center", fontsize=6,
                    color=color if ldt else "0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], ["90%", "99%", "99.9%", "99.99%", "99.999%"])
    ax.minorticks_off()
    ax.set_ylim(0.75, 1.2e-6)
    ax.grid(True, axis="y", color="0.9", lw=0.6)
    ax.set_xlabel("Training + solving compute (PFLOPs)")
    ax.set_ylabel("Accuracy")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    fig.tight_layout()
    return pdf_bytes(fig)


def main() -> None:
    args = cli("Figure 1(b) (Sudoku-Extreme compute frontier).", default_manifest="table1_sudoku")
    print(f"wrote {write_output('sudoku_compute_frontier.pdf', plot(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
