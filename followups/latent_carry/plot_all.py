"""Plot E6 in-training solve accuracy curves collected by collect.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from followups.latent_carry.configs import CONFIGS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
COLORS = {
    "baseline": "#4C78A8",
    "carry_h": "#F58518",
    "carry_z": "#54A24B",
    "zero_carry_z": "#E45756",
}


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    found = 0
    for config, spec in CONFIGS.items():
        curves: list[list[dict]] = []
        for seed in range(spec["n_seeds"]):
            path = RESULTS / f"{config}_seed{seed}.train_curve.jsonl"
            if not path.exists():
                continue
            curve = [json.loads(line) for line in path.read_text().splitlines() if line]
            curves.append(curve)
            ax.plot(
                [row["step"] for row in curve],
                [row["correct"] / 200 for row in curve],
                color=COLORS[config],
                alpha=0.22,
                linewidth=1,
            )
        if not curves:
            continue
        found += len(curves)
        common_steps = sorted(set.intersection(*[
            {row["step"] for row in curve} for curve in curves
        ]))
        by_curve = [{row["step"]: row for row in curve} for curve in curves]
        means = [
            sum(rows[step]["correct"] / 200 for rows in by_curve) / len(by_curve)
            for step in common_steps
        ]
        ax.plot(
            common_steps, means, color=COLORS[config], linewidth=2.4,
            label=f"{config} (n={len(curves)})",
        )

    if not found:
        raise SystemExit("No collected learning curves; run collect.py first")
    ax.set(xlabel="Training step", ylabel="Mini-solve accuracy", ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = PLOTS / "learning_curves.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
