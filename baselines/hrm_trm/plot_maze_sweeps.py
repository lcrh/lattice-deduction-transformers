"""Plot exact and lenient Maze-Hard accuracy over training, with and without augmentation.

Reads the per-checkpoint sweep JSONs written by
{trm,hrm}_eval/modal_eval.py::sweep (lists of {step, n, exact, lenient}) and
renders one PNG with a panel per model. Run by the eval_*.sh scripts, or:

    uv run python baselines/hrm_trm/plot_maze_sweeps.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Sweep JSON per model and training condition.
PANELS = {
    "TRM": {"aug": "trm_aug_sweep_lcycles4.json", "no-aug": "trm_noaug_sweep_lcycles4.json"},
    "HRM": {"aug": "hrm_aug_sweep.json", "no-aug": "hrm_noaug_sweep.json"},
}
COND_COLOR = {"aug": "#c0563b", "no-aug": "#2f6f9f"}


def _load(name):
    p = os.path.join(RESULTS_DIR, name)
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
out = os.path.join(RESULTS_DIR, "maze_aug_vs_noaug.png")
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
