"""Plot exact vs lenient maze-hard accuracy over training, aug vs no-aug.

Reads the per-checkpoint sweep JSONs produced by
{trm,hrm}_eval/modal_eval.py::sweep  (lists of {step, n, exact, lenient}) and
renders one PNG with a panel per model. Run via the eval_*.sh scripts, or:

    uv run --with matplotlib python experiments/plot_maze_sweeps.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTES = os.path.join(os.path.dirname(__file__), "results")

# released-checkpoint reference lines (our lenient eval of the public weights)
REF = {
    "TRM": {"exact": 83.8, "lenient": 90.7},
    "HRM": {"exact": 74.4, "lenient": 81.6},
}
# (model, sweep-json basename) per condition
PANELS = {
    "TRM": {"aug": "trm_aug_sweep.json", "no-aug": "trm_noaug_sweep.json"},
    "HRM": {"aug": "hrm_aug_sweep.json", "no-aug": "hrm_noaug_sweep.json"},
}
COND_COLOR = {"aug": "#c0563b", "no-aug": "#2f6f9f"}


def _load(name):
    p = os.path.join(NOTES, name)
    if not os.path.exists(p):
        return None
    rows = sorted(json.load(open(p)), key=lambda r: r["step"])
    xs = [r["step"] for r in rows]
    ex = [100 * r["exact"] / r["n"] for r in rows]
    le = [100 * r["lenient"] / r["n"] for r in rows]
    return xs, ex, le


fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 5.6), sharey=True)
fig.patch.set_facecolor("white")

for ax, (model, conds) in zip(axes, PANELS.items()):
    for cond, fname in conds.items():
        d = _load(fname)
        if d is None:
            continue
        xs, ex, le = d
        c = COND_COLOR[cond]
        ax.plot(xs, le, "-", color=c, lw=2.2, label=f"{cond} · lenient")
        ax.plot(xs, ex, "--", color=c, lw=1.6, alpha=0.8, label=f"{cond} · exact")
    # released reference lines
    ax.axhline(REF[model]["lenient"], color="#777", lw=1, ls=":")
    ax.axhline(REF[model]["exact"], color="#bbb", lw=1, ls=":")
    ax.text(0.99, REF[model]["lenient"] + 1, f"released lenient {REF[model]['lenient']}%",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=7.5, color="#777")
    ax.text(0.99, REF[model]["exact"] - 3.5, f"released exact {REF[model]['exact']}%",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=7.5, color="#999")
    ax.set_title(model, fontsize=14, fontweight="bold", color="#15233b")
    ax.set_xlabel("training step")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="center right")

axes[0].set_ylabel("maze-hard test accuracy (%)")
fig.suptitle("Maze-30×30-hard: exact vs lenient accuracy over training "
             "(8× dihedral augmentation vs none)",
             fontsize=14, fontweight="bold", color="#15233b", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(NOTES, "maze_aug_vs_noaug.png")
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
