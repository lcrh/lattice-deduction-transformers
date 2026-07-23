"""E1 figures D1, D2, D3, D4 from results/summary.csv (+ staged train curves).

    uv run --with matplotlib python followups/arch_ablation/plot_all.py

Reads:
  results/summary.csv                               (from collect.py)
  results/curves/<config>_seed<N>.train_curve.jsonl (staged by collect.py)

Writes under plots/:
  d1_loops.pdf            solve rate + unsound-elim rate vs per-forward FLOPs (log-x)
  d1_escalation_curves.pdf  C2 learning curves (solve count vs training FLOPs, log-x)
  d2_curves.pdf           in-train solve count vs step: baseline vs d2_final_only
  d3_ce_curves.pdf        in-train solve count vs step: d3_ce0 / baseline / d3_ce1
  d4_soundness.pdf        unsound rate / wrong / solve rate / calls-per-solve vs ratio

Degrades gracefully: a config with no data is skipped (its series omitted) and
the rest still render. Prints which figures were written and which series were
skipped for lack of data.

FLOPs axis (relative proxy, documented): per-forward FLOPs are approximated as
    layers * dim^2 * n_loops
(the feed-forward term; the attention-score term ~ layers*dim*L is a
second-order deviation). This is a RELATIVE axis — exactness is not required.
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Absolute package import — scripts run from the repo root.
from followups.arch_ablation import configs as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CURVES_DIR = os.path.join(RESULTS_DIR, "curves")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
PLOTS_DIR = os.path.join(HERE, "plots")

# House style (cf. repro/plot_maze_sweeps.py).
INK = "#15233b"
C_MAIN = "#c0563b"     # C1 tied loop line
C_C3 = "#2f6f9f"       # C3 shapes
C_C4 = "#4a8b5c"       # C4 untied
C_C2 = "#8a5fb0"       # C2 escalation
C_BASE = "#777777"     # baseline reference
SERIES_COLORS = ["#c0563b", "#2f6f9f", "#4a8b5c", "#8a5fb0", "#d09a1e", "#555555"]

skipped_series: list[str] = []
written: list[str] = []


# --------------------------------------------------------------------------
# Data loading.
# --------------------------------------------------------------------------

def _fnum(v):
    """Parse a CSV cell to float, or None if blank / non-numeric / range str."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_summary() -> dict[str, dict[str, list]]:
    """Return {config: {"seeds": [row,...], "mean": row|None}} from summary.csv.

    Only per-seed rows (numeric seed) go in "seeds"; the seed=="mean" row (if
    present) goes in "mean". Missing file -> empty dict.
    """
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot] no {SUMMARY_CSV} — run collect.py first. Nothing to plot.",
              flush=True)
        return {}
    out: dict[str, dict] = {}
    with open(SUMMARY_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            cfg = row["config"]
            entry = out.setdefault(cfg, {"seeds": [], "mean": None})
            if row["seed"] == "mean":
                entry["mean"] = row
            elif row["seed"] == "range":
                continue
            else:
                entry["seeds"].append(row)
    return out


def load_curve(config: str, seed: int) -> list[dict] | None:
    """Load a staged train_curve.jsonl -> list of {step, correct, ...} dicts."""
    path = os.path.join(CURVES_DIR, f"{config}_seed{seed}.train_curve.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows or None


def fwd_flops(config: str) -> float:
    """Relative per-forward FLOPs proxy = layers * dim^2 * n_loops."""
    eff = C.effective_flags(config)
    return float(eff["num_layers"]) * float(eff["dim"]) ** 2 * float(eff["n_loops"])


def train_flops(config: str) -> float:
    """Relative training FLOPs proxy = per-forward FLOPs * steps."""
    eff = C.effective_flags(config)
    return fwd_flops(config) * float(eff["steps"])


def _mean_range(rows: list[dict], col: str):
    """Return (mean, lo, hi) over per-seed rows for a numeric column, or None."""
    vals = [_fnum(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals), min(vals), max(vals)


def _yerr(m: float, lo: float, hi: float):
    """Non-negative asymmetric yerr for errorbar (clamps tiny float negatives).

    matplotlib raises if yerr contains any negative value; a single-seed point
    (lo==hi==m) or float rounding can otherwise produce a tiny negative.
    """
    return [[max(0.0, m - lo)], [max(0.0, hi - m)]]


# --------------------------------------------------------------------------
# Figure D1: solve rate + unsound-elim rate vs per-forward FLOPs.
# --------------------------------------------------------------------------

def fig_d1(data: dict) -> None:
    C1 = [f"d1_L{L}" for L in (1, 2, 4, 8, 16, 32)]
    C3 = ["d1_L1", "d1_shape8x92", "d1_shape16x64", "d1_shape32x44"]
    C4 = ["d1_untied8", "d1_untied16", "d1_wide", "d1_untied16_max"]
    C2 = ["d1_L2_cm", "d1_L1_cm", "d1_L1_cm4x", "d1_L1_bigdata"]

    def series(names, col):
        pts = []
        for name in names:
            entry = data.get(name)
            if not entry or not entry["seeds"]:
                skipped_series.append(f"d1_loops[{col}]:{name}")
                continue
            mr = _mean_range(entry["seeds"], col)
            if mr is None:
                skipped_series.append(f"d1_loops[{col}]:{name}")
                continue
            m, lo, hi = mr
            pts.append((name, fwd_flops(name), m, lo, hi))
        return pts

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
    fig.patch.set_facecolor("white")

    def plot_panel(ax, col, ylabel):
        # C1 main line.
        c1 = series(C1, col)
        if c1:
            c1.sort(key=lambda p: p[1])
            xs = [p[1] for p in c1]
            ys = [p[2] for p in c1]
            los = [p[3] for p in c1]
            his = [p[4] for p in c1]
            ax.plot(xs, ys, "-o", color=C_MAIN, lw=2.2, label="C1 tied loop sweep")
            ax.fill_between(xs, los, his, color=C_MAIN, alpha=0.15)
        # C3 shapes clustered at L=1 FLOPs, annotated by shape.
        for name, x, m, lo, hi in series(C3, col):
            if name == "d1_L1":
                continue  # already the C1 L=1 point
            ax.errorbar(x, m, yerr=_yerr(m, lo, hi), fmt="s",
                        color=C_C3, ms=7, capsize=3)
            ax.annotate(name.replace("d1_shape", ""), (x, m),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=C_C3)
        # C4 untied — marker size ~ params.
        c4 = series(C4, col)
        if c4:
            params = [C.effective_flags(n)["num_layers"] * C.effective_flags(n)["dim"] ** 2
                      for n, *_ in c4]
            pmin = min(params) or 1
            for (name, x, m, lo, hi), p in zip(c4, params):
                ax.errorbar(x, m, yerr=_yerr(m, lo, hi), fmt="^",
                            color=C_C4, ms=6 + 6 * (p / pmin) ** 0.25, capsize=3,
                            alpha=0.85)
                ax.annotate(name.replace("d1_", ""), (x, m),
                            textcoords="offset points", xytext=(6, -10),
                            fontsize=7, color=C_C4)
        # C2 escalation — open markers.
        for name, x, m, lo, hi in series(C2, col):
            ax.errorbar(x, m, yerr=_yerr(m, lo, hi), fmt="o",
                        mfc="none", mec=C_C2, ms=8, capsize=3)
            ax.annotate(name.replace("d1_", ""), (x, m),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color=C_C2)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    plot_panel(ax0, "accuracy", "solve rate (correct / n)")
    plot_panel(ax1, "unsound_rate", "unsound-elimination rate")
    ax1.set_xlabel("per-forward FLOPs (relative: layers · dim² · L, log)")
    ax0.legend(fontsize=8.5, loc="best")
    ax0.set_title("D1 — recursion vs. depth vs. params", fontsize=13,
                  fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "d1_loops.pdf")


# --------------------------------------------------------------------------
# Figure D1 companion: C2 escalation learning curves.
# --------------------------------------------------------------------------

def fig_d1_escalation(data: dict) -> None:
    C2 = ["d1_L2_cm", "d1_L1_cm", "d1_L1_cm4x", "d1_L1_bigdata"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")
    any_series = False

    # Baseline curve overlaid (seed 0 if present, else any).
    for base_seed in range(3):
        bc = load_curve("baseline", base_seed)
        if bc:
            fpf = fwd_flops("baseline")
            xs = [r["step"] * fpf for r in bc]
            ys = [r["correct"] for r in bc]
            ax.plot(xs, ys, "-", color=C_BASE, lw=2.0, alpha=0.9,
                    label="baseline (tied 4×16)")
            any_series = True
            break
    else:
        skipped_series.append("d1_escalation:baseline")

    for i, name in enumerate(C2):
        fpf = fwd_flops(name)
        drawn = False
        for seed in range(C.CONFIGS[name]["n_seeds"]):
            cur = load_curve(name, seed)
            if not cur:
                continue
            xs = [r["step"] * fpf for r in cur]
            ys = [r["correct"] for r in cur]
            ax.plot(xs, ys, "-", color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    lw=1.8, alpha=0.85,
                    label=name.replace("d1_", "") if not drawn else None)
            drawn = True
            any_series = True
        if not drawn:
            skipped_series.append(f"d1_escalation:{name}")

    if not any_series:
        plt.close(fig)
        skipped_series.append("d1_escalation:ALL(no curves)")
        return
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs (relative: per-forward · step, log)")
    ax.set_ylabel("in-train solve count (max_rounds 5)")
    ax.set_title("D1-C2 — escalation learning curves (plateau vs. still-climbing)",
                 fontsize=12, fontweight="bold", color=INK)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    _save(fig, "d1_escalation_curves.pdf")


# --------------------------------------------------------------------------
# In-train solve-count-vs-step curves (D2, D3).
# --------------------------------------------------------------------------

def _curve_fig(configs_colors: list[tuple[str, str]], title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")
    any_series = False
    for name, color in configs_colors:
        if name not in C.CONFIGS:
            continue
        labelled = False
        for seed in range(C.CONFIGS[name]["n_seeds"]):
            cur = load_curve(name, seed)
            if not cur:
                continue
            xs = [r["step"] for r in cur]
            ys = [r["correct"] for r in cur]
            ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.8,
                    label=name if not labelled else None)
            labelled = True
            any_series = True
        if not labelled:
            skipped_series.append(f"{fname}:{name}")
    if not any_series:
        plt.close(fig)
        skipped_series.append(f"{fname}:ALL(no curves)")
        return
    ax.set_xlabel("training step")
    ax.set_ylabel("in-train solve count (max_rounds 5)")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    _save(fig, fname)


def fig_d2() -> None:
    _curve_fig(
        [("baseline", C_BASE), ("d2_final_only", C_MAIN)],
        "D2 — deep supervision: in-train solve count vs. step",
        "d2_curves.pdf",
    )


def fig_d3() -> None:
    _curve_fig(
        [("d3_ce0", C_C3), ("baseline", C_BASE), ("d3_ce1", C_MAIN)],
        "D3 — L_CE weight: in-train solve count vs. step",
        "d3_ce_curves.pdf",
    )


# --------------------------------------------------------------------------
# Figure D4: soundness knobs vs asymmetry ratio.
# --------------------------------------------------------------------------

def fig_d4(data: dict) -> None:
    # Ratio-ordered configs (bce_pos/bce_neg). d4_nocls handled separately.
    ratio_configs = [
        ("d4_sym", 1.0),
        ("d4_ratio2", 2.0),
        ("baseline", 8.0),
        ("d4_ratio32", 32.0),
    ]
    panels = [
        ("unsound_rate", "unsound-elim rate"),
        ("wrong", "wrong-answer count"),
        ("accuracy", "solve rate"),
        ("calls_per_solve", "calls / solve"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("white")
    any_data = False

    for ax, (col, ylabel) in zip(axes.flat, panels):
        xs, ys, los, his = [], [], [], []
        for name, ratio in ratio_configs:
            entry = data.get(name)
            if not entry or not entry["seeds"]:
                skipped_series.append(f"d4[{col}]:{name}")
                continue
            mr = _mean_range(entry["seeds"], col)
            if mr is None:
                skipped_series.append(f"d4[{col}]:{name}")
                continue
            m, lo, hi = mr
            xs.append(ratio); ys.append(m); los.append(lo); his.append(hi)
        if xs:
            any_data = True
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            los = [los[i] for i in order]; his = [his[i] for i in order]
            yerr = [[max(0.0, y - l) for y, l in zip(ys, los)],
                    [max(0.0, h - y) for y, h in zip(ys, his)]]
            ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color=C_MAIN, capsize=3, lw=2.0)
        # d4_nocls as a separate annotated point (plotted off-axis at ratio 8).
        nocls = data.get("d4_nocls")
        if nocls and nocls["seeds"]:
            mr = _mean_range(nocls["seeds"], col)
            if mr is not None:
                m, lo, hi = mr
                ax.errorbar([8.0], [m], yerr=_yerr(m, lo, hi), fmt="D",
                            color=C_C4, ms=9, capsize=3, label="d4_nocls (no CLS head)")
                ax.annotate("nocls", (8.0, m), textcoords="offset points",
                            xytext=(8, 6), fontsize=7.5, color=C_C4)
                any_data = True
        ax.set_xscale("log")
        ax.set_xlabel("BCE asymmetry ratio (w+/w−)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if col == "unsound_rate":
            ax.legend(fontsize=8, loc="best")

    if not any_data:
        plt.close(fig)
        skipped_series.append("d4_soundness:ALL(no data)")
        return
    fig.suptitle("D4 — soundness-pressure knobs vs. asymmetry ratio",
                 fontsize=13, fontweight="bold", color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "d4_soundness.pdf")


# --------------------------------------------------------------------------

def _save(fig, fname: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    written.append(path)


def main() -> None:
    data = load_summary()
    # D2/D3 rely on curves only, so they run even if summary.csv is empty.
    fig_d1(data)
    fig_d1_escalation(data)
    fig_d2()
    fig_d3()
    fig_d4(data)

    print("\n=== plot_all summary ===", flush=True)
    if written:
        print("Figures written:", flush=True)
        for p in written:
            print(f"  {p}", flush=True)
    else:
        print("No figures written (no data found).", flush=True)
    if skipped_series:
        print("Series skipped for lack of data:", flush=True)
        for s in skipped_series:
            print(f"  {s}", flush=True)


if __name__ == "__main__":
    main()
