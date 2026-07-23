"""E4 (ood_snowflake) figure — accuracy & soundness vs. test order.

Reads `results/per_order.csv` (written by collect.py) and renders the
order-transfer figure the README asks for: per training range, accuracy and
soundness (wrong-answer rate) as a function of test order, with a vertical
dashed line at the OOD boundary (max train order + 0.5).

Layout: one column per config (e4_all / e4_leq5 / e4_leq6 [/ e4_leq5_rope]),
two stacked rows:
  - top row:    accuracy vs test order (mean over seeds, shaded min-max band)
  - bottom row: soundness = wrong-answer rate vs test order (same band), so an
    accuracy drop that PRESERVES soundness (failures = abstentions/timeouts)
    is visually distinct from one that emits wrong answers.

Only the per-seed rows (seed is an integer) are used; mean/min/max are
recomputed here so the band is over accuracy AND wrong-rate consistently.

Degrades gracefully: configs/orders with no data are skipped (series omitted),
the rest still render, and a summary of what was drawn/skipped is printed.

Usage:
    uv run --with matplotlib python followups/ood_snowflake/plot_order_transfer.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
PLOTS_DIR = _HERE / "plots"

# House style (mirrors repro/plot_maze_sweeps.py).
ACC_COLOR = "#2f6f9f"      # accuracy line (blue)
SOUND_COLOR = "#c0563b"    # wrong-rate line (red)
INK = "#15233b"
# Config panel order + friendly titles.
PANEL_ORDER = ["e4_all", "e4_leq5", "e4_leq6", "e4_leq5_rope"]
PANEL_TITLE = {
    "e4_all": "e4_all  (train 4-8, control)",
    "e4_leq5": "e4_leq5  (train {4,5})",
    "e4_leq6": "e4_leq6  (train {4,5,6})",
    "e4_leq5_rope": "e4_leq5_rope  (train {4,5}, RoPE)",
}


def _load(csv_path: Path):
    """Return nested dict: config -> {"boundary": int, order -> {metric: [vals]}}.

    Only per-seed rows (integer `seed`) are consumed. `accuracy` and the wrong
    rate (wrong / (correct+wrong+timeout)) are collected per (config, order)
    across seeds.
    """
    data: dict[str, dict] = defaultdict(
        lambda: {"boundary": None, "orders": defaultdict(
            lambda: {"acc": [], "wrong_rate": []})})
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            seed = row["seed"]
            if not seed.lstrip("-").isdigit():
                continue  # skip mean/min/max roll-up rows
            cfg = row["config"]
            order = int(row["test_order"])
            correct = int(row["correct"])
            wrong = int(row["wrong"])
            timeout = int(row["timeout"])
            denom = correct + wrong + timeout
            if denom <= 0:
                continue
            acc = float(row["accuracy"])
            wrong_rate = wrong / denom
            data[cfg]["boundary"] = int(row["train_max_order"])
            od = data[cfg]["orders"][order]
            od["acc"].append(acc)
            od["wrong_rate"].append(wrong_rate)
    return data


def _series(order_map, metric):
    """Return (orders, mean, lo, hi) sorted by order for a metric."""
    orders = sorted(order_map)
    xs, mean, lo, hi = [], [], [], []
    for o in orders:
        vals = order_map[o][metric]
        if not vals:
            continue
        xs.append(o)
        m = sum(vals) / len(vals)
        mean.append(m)
        lo.append(min(vals))
        hi.append(max(vals))
    return xs, mean, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(RESULTS_DIR / "per_order.csv"))
    ap.add_argument("--out", default=str(PLOTS_DIR / "order_transfer.pdf"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run collect.py first.",
              file=sys.stderr)
        sys.exit(1)

    data = _load(csv_path)
    present = [c for c in PANEL_ORDER if c in data and data[c]["orders"]]
    # Include any config in the CSV not covered by PANEL_ORDER (future-proof).
    present += [c for c in data if c not in PANEL_ORDER and data[c]["orders"]]

    skipped = [c for c in PANEL_ORDER if c not in present]
    if not present:
        print("ERROR: no plottable per-seed data in the CSV (all configs empty).",
              file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(present)
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 7.2), squeeze=False,
                             sharex="col")
    fig.patch.set_facecolor("white")

    drawn_orders: dict[str, list[int]] = {}
    for j, cfg in enumerate(present):
        boundary = data[cfg]["boundary"]
        order_map = data[cfg]["orders"]
        ax_acc = axes[0][j]
        ax_snd = axes[1][j]

        xa, ma, la, ha = _series(order_map, "acc")
        xs, ms, ls, hs = _series(order_map, "wrong_rate")
        drawn_orders[cfg] = xa

        # Accuracy (top).
        if xa:
            ax_acc.fill_between(xa, la, ha, color=ACC_COLOR, alpha=0.18, lw=0)
            ax_acc.plot(xa, ma, "-o", color=ACC_COLOR, lw=2.2, ms=5,
                        label="accuracy")
        # Soundness = wrong-answer rate (bottom).
        if xs:
            ax_snd.fill_between(xs, ls, hs, color=SOUND_COLOR, alpha=0.18, lw=0)
            ax_snd.plot(xs, ms, "-o", color=SOUND_COLOR, lw=2.2, ms=5,
                        label="wrong-answer rate")

        # OOD boundary line at max_train_order + 0.5.
        if boundary is not None:
            for ax in (ax_acc, ax_snd):
                ax.axvline(boundary + 0.5, color="#555", lw=1.3, ls="--")
            ax_acc.text(boundary + 0.55, 0.02, "OOD ->", fontsize=8,
                        color="#555", ha="left", va="bottom",
                        transform=ax_acc.get_xaxis_transform())

        ax_acc.set_title(PANEL_TITLE.get(cfg, cfg), fontsize=12,
                         fontweight="bold", color=INK)
        ax_acc.set_ylim(0, 1.02)
        ax_snd.set_ylim(0, max(0.05, (max(hs) * 1.25) if hs else 0.05))
        for ax in (ax_acc, ax_snd):
            ax.grid(True, alpha=0.25)
        ax_snd.set_xlabel("test order")
        # Integer x ticks over the drawn orders.
        all_x = sorted(set(xa) | set(xs))
        if all_x:
            for ax in (ax_acc, ax_snd):
                ax.set_xticks(all_x)

    axes[0][0].set_ylabel("test accuracy")
    axes[1][0].set_ylabel("wrong-answer rate\n(lower = sounder)")
    axes[0][0].legend(fontsize=8.5, loc="lower left")
    axes[1][0].legend(fontsize=8.5, loc="upper left")

    fig.suptitle("E4 Snowflake order transfer: accuracy & soundness vs. test order\n"
                 "(mean +/- min-max over seeds; dashed line = OOD boundary)",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, facecolor="white")
    print(f"Wrote {out_path}")
    for cfg in present:
        print(f"  drew {cfg}: orders {drawn_orders.get(cfg, [])}")
    if skipped:
        print(f"  skipped (no data): {skipped}")


if __name__ == "__main__":
    main()
