"""E3 figures O1a / O1b / O2 / O3 / O4 from results/ (collect.py output).

    uv run --with matplotlib python followups/deduction_operator/plot_all.py

Reads:
  results/summary.csv        (from collect.py)
  results/profile_iters.csv  (staged by collect.py; O2)

Writes under plots/:
  o1a_loop_scaling.pdf    solve + unsound rate vs L_eval (log-x), plus a
                          FLOPs-normalized companion panel (solve rate vs
                          total forward-FLOPs/puzzle).
  o1b_transfer.pdf        heatmap of solve rate over (L_train, L_eval), with
                          the d2_final_only row appended below the matrix.
  o2_per_iteration.pdf    per-iteration eliminations (cumulative + marginal),
                          unsound rate, and CLS logit trajectory (SAT vs UNSAT).
  o3_per_pass_compounding.pdf  unsound-elimination rate vs deduce pass index
                          for the multi-pass O3 configs (deduce-to-fixpoint).
  o4_thresholds.pdf       theta_elim sensitivity (baseline + d4_sym overlaid)
                          and theta_CLS sensitivity (baseline).

Degrades gracefully: a figure with no data is skipped and the rest render.

FLOPs axis (relative proxy, documented): per-forward FLOPs are approximated as
proportional to n_loops (the tied recurrence dominates; layers/dim are fixed
across every E3 eval since they all consume the same-shape backbone). The
O1a companion x-axis uses `effective_n_loops` as the relative forward-FLOPs
proxy — "more loops" and "more search rounds" then compete on one axis.
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from followups.deduction_operator import configs as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
PROFILE_CSV = os.path.join(RESULTS_DIR, "profile_iters.csv")
O3_PER_PASS_CSV = os.path.join(RESULTS_DIR, "o3_per_pass.csv")
PLOTS_DIR = os.path.join(HERE, "plots")

INK = "#15233b"
C_MAIN = "#c0563b"
C_ALT = "#2f6f9f"
C_UNSAT = "#b0483b"
C_SAT = "#2f6f9f"
STRAT_COLORS = {"early": "#4a8b5c", "mid": "#d09a1e", "late": "#8a5fb0"}

written: list[str] = []
skipped: list[str] = []


def _fnum(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_summary() -> dict[str, dict]:
    """Return {config: {"seeds": [row,...], "mean": row|None}}."""
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot] no {SUMMARY_CSV} — run collect.py first.", flush=True)
        return {}
    out: dict[str, dict] = {}
    with open(SUMMARY_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            entry = out.setdefault(row["config"], {"seeds": [], "mean": None})
            if row["seed"] == "mean":
                entry["mean"] = row
            elif row["seed"] == "range":
                continue
            else:
                entry["seeds"].append(row)
    return out


def _mean(rows: list[dict], col: str):
    vals = [_fnum(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _range(rows: list[dict], col: str):
    vals = [_fnum(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return min(vals), max(vals)


def _save(fig, fname: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    written.append(path)


# --------------------------------------------------------------------------
# O1a — baseline loop scaling.
# --------------------------------------------------------------------------

def fig_o1a(data: dict) -> None:
    Ls = list(C.L_EVAL_AXIS)
    pts = []  # (L, acc, acc_lo, acc_hi, unsound)
    for L in Ls:
        name = f"o1_scale_L{L}"
        entry = data.get(name)
        if not entry or not entry["seeds"]:
            skipped.append(f"o1a:{name}")
            continue
        acc = _mean(entry["seeds"], "accuracy")
        uns = _mean(entry["seeds"], "unsound_rate")
        rng = _range(entry["seeds"], "accuracy")
        if acc is None:
            skipped.append(f"o1a:{name}")
            continue
        lo, hi = rng if rng else (acc, acc)
        pts.append((L, acc, lo, hi, uns))
    if not pts:
        skipped.append("o1a:ALL(no data)")
        return
    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]
    acc = [p[1] for p in pts]
    los = [p[2] for p in pts]
    his = [p[3] for p in pts]
    uns = [p[4] if p[4] is not None else float("nan") for p in pts]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # Left: solve + unsound vs L_eval (log-x).
    ax0.plot(xs, acc, "-o", color=C_MAIN, lw=2.2, label="solve rate")
    ax0.fill_between(xs, los, his, color=C_MAIN, alpha=0.15)
    ax0b = ax0.twinx()
    ax0b.plot(xs, uns, "--s", color=C_ALT, lw=1.8, label="unsound rate")
    ax0.set_xscale("log", base=2)
    ax0.set_xlabel("L_eval (internal loops, log2)")
    ax0.set_ylabel("solve rate (correct / n)", color=C_MAIN)
    ax0b.set_ylabel("unsound-elimination rate", color=C_ALT)
    ax0.set_xticks(xs); ax0.set_xticklabels([str(x) for x in xs])
    ax0.grid(True, alpha=0.25)
    ax0.set_title("O1a — baseline loop scaling", fontsize=12,
                  fontweight="bold", color=INK)

    # Right: FLOPs-normalized companion — solve rate vs relative forward-FLOPs
    # (proxy proportional to L_eval, since layers/dim fixed across E3 evals).
    ax1.plot(xs, acc, "-o", color=C_MAIN, lw=2.2)
    ax1.fill_between(xs, los, his, color=C_MAIN, alpha=0.15)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("relative forward-FLOPs / puzzle  (proxy ∝ L_eval, log2)")
    ax1.set_ylabel("solve rate")
    ax1.set_xticks(xs); ax1.set_xticklabels([str(x) for x in xs])
    ax1.grid(True, alpha=0.25)
    ax1.set_title("O1a — compute-honest view", fontsize=12,
                  fontweight="bold", color=INK)

    fig.tight_layout()
    _save(fig, "o1a_loop_scaling.pdf")


# --------------------------------------------------------------------------
# O1b — transfer heatmap.
# --------------------------------------------------------------------------

def fig_o1b(data: dict) -> None:
    Lt_axis = list(C.L_TRAIN_AXIS)
    Le_axis = list(C.L_EVAL_AXIS)
    mat = np.full((len(Lt_axis), len(Le_axis)), np.nan)
    for i, Lt in enumerate(Lt_axis):
        for j, Le in enumerate(Le_axis):
            entry = data.get(f"o1_xfer_Lt{Lt}_Le{Le}")
            if entry and entry["seeds"]:
                acc = _mean(entry["seeds"], "accuracy")
                if acc is not None:
                    mat[i, j] = acc
    # d2_final_only row.
    d2_row = np.full(len(Le_axis), np.nan)
    for j, Le in enumerate(Le_axis):
        entry = data.get(f"o1_d2fo_Le{Le}")
        if entry and entry["seeds"]:
            acc = _mean(entry["seeds"], "accuracy")
            if acc is not None:
                d2_row[j] = acc

    if np.isnan(mat).all() and np.isnan(d2_row).all():
        skipped.append("o1b:ALL(no data)")
        return

    # Stack: transfer matrix on top, a gap, then the d2_final_only row.
    full = np.vstack([mat, np.full((1, len(Le_axis)), np.nan), d2_row[None, :]])
    row_labels = [f"L_train={Lt}" for Lt in Lt_axis] + ["", "d2_final_only"]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(full, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(Le_axis)))
    ax.set_xticklabels([str(Le) for Le in Le_axis])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("L_eval")
    ax.set_title("O1b — train/eval loop transfer (solve rate)",
                 fontsize=12, fontweight="bold", color=INK)
    # Annotate cells.
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            v = full[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="solve rate")
    fig.tight_layout()
    _save(fig, "o1b_transfer.pdf")


# --------------------------------------------------------------------------
# O2 — per-iteration profile (from profile_iters.csv).
# --------------------------------------------------------------------------

def load_profile() -> list[dict] | None:
    if not os.path.exists(PROFILE_CSV):
        return None
    rows = []
    with open(PROFILE_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows or None


def fig_o2() -> None:
    rows = load_profile()
    if not rows:
        skipped.append("o2:ALL(no profile csv)")
        return
    strata = sorted({r["stratum"] for r in rows})
    # Group by stratum -> list of (iter, cum, marg, cum_uns, marg_uns).
    by_strat: dict[str, list] = {s: [] for s in strata}
    cls_sat: dict[int, float] = {}
    cls_unsat: dict[int, float] = {}
    for r in rows:
        it = int(r["iter"])
        by_strat[r["stratum"]].append((
            it, float(r["cum_elim_per_state"]), float(r["marg_elim_per_state"]),
            float(r["cum_unsound_per_state"]), float(r["marg_unsound_per_state"]),
        ))
        cls_sat[it] = float(r["cls_logit_sat_mean"])
        cls_unsat[it] = float(r["cls_logit_unsat_mean"])
    for s in by_strat:
        by_strat[s].sort(key=lambda t: t[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor("white")
    ax_elim, ax_uns, ax_cls = axes

    for s in strata:
        col = STRAT_COLORS.get(s, "#555555")
        d = by_strat[s]
        its = [t[0] for t in d]
        cum = [t[1] for t in d]
        marg = [t[2] for t in d]
        ax_elim.plot(its, cum, "-", color=col, lw=2.0, label=f"{s} (cumulative)")
        ax_elim.plot(its, marg, "--", color=col, lw=1.2, alpha=0.7,
                     label=f"{s} (marginal)")
        # unsound rate = cum_unsound / max(cum_elim, eps).
        cum_uns = [t[3] for t in d]
        rate = [(u / e if e > 1e-9 else 0.0) for u, e in zip(cum_uns, cum)]
        ax_uns.plot(its, rate, "-", color=col, lw=2.0, label=s)

    ax_elim.set_xlabel("iteration ℓ")
    ax_elim.set_ylabel("eliminations / state")
    ax_elim.set_title("O2 — eliminations per forward", fontsize=11,
                      fontweight="bold", color=INK)
    ax_elim.grid(True, alpha=0.25)
    ax_elim.legend(fontsize=7.5, loc="best")

    ax_uns.set_xlabel("iteration ℓ")
    ax_uns.set_ylabel("unsound fraction (cumulative)")
    ax_uns.set_title("O2 — unsound rate vs ℓ", fontsize=11,
                     fontweight="bold", color=INK)
    ax_uns.grid(True, alpha=0.25)
    ax_uns.legend(fontsize=8, loc="best")

    its = sorted(cls_sat)
    ax_cls.plot(its, [cls_sat[i] for i in its], "-", color=C_SAT, lw=2.0,
                label="SAT states")
    ax_cls.plot(its, [cls_unsat[i] for i in its], "-", color=C_UNSAT, lw=2.0,
                label="UNSAT states")
    ax_cls.set_xlabel("iteration ℓ")
    ax_cls.set_ylabel("CLS conflict logit (mean)")
    ax_cls.set_title("O2 — CLS logit trajectory", fontsize=11,
                     fontweight="bold", color=INK)
    ax_cls.grid(True, alpha=0.25)
    ax_cls.legend(fontsize=8, loc="best")

    fig.tight_layout()
    _save(fig, "o2_per_iteration.pdf")


# --------------------------------------------------------------------------
# O4 — threshold sensitivity.
# --------------------------------------------------------------------------

def _tag(x):
    return str(x).replace("0.", "0").replace(".", "")


def fig_o4(data: dict) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    any_data = False

    def elim_series(prefix, axis, color, label):
        nonlocal any_data
        xs, acc, wrong = [], [], []
        for th in axis:
            entry = data.get(f"{prefix}_{_tag(th)}")
            if not entry or not entry["seeds"]:
                continue
            a = _mean(entry["seeds"], "accuracy")
            w = _mean(entry["seeds"], "wrong")
            if a is None:
                continue
            xs.append(th); acc.append(a); wrong.append(w if w is not None else float("nan"))
        if not xs:
            skipped.append(f"o4:{prefix}")
            return
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]; acc = [acc[i] for i in order]
        wrong = [wrong[i] for i in order]
        ax0.plot(xs, acc, "-o", color=color, lw=2.0, label=f"{label} solve")
        any_data = True
        return xs, wrong

    # theta_elim panel: baseline + d4_sym solve rate, plus wrong-answer counts
    # on a twin axis.
    b = elim_series("o4_elim_base", C.THETA_ELIM_AXIS, C_MAIN, "baseline")
    s = elim_series("o4_elim_sym", C.THETA_ELIM_AXIS, C_ALT, "d4_sym")
    ax0b = ax0.twinx()
    if b:
        ax0b.plot(b[0], b[1], "--^", color=C_MAIN, lw=1.2, alpha=0.6,
                  label="baseline wrong")
    if s:
        ax0b.plot(s[0], s[1], "--^", color=C_ALT, lw=1.2, alpha=0.6,
                  label="d4_sym wrong")
    ax0.set_xscale("log")
    ax0.set_xlabel("θ_elim")
    ax0.set_ylabel("solve rate")
    ax0b.set_ylabel("wrong-answer count")
    ax0.set_title("O4 — θ_elim sensitivity (baseline vs d4_sym)",
                  fontsize=11, fontweight="bold", color=INK)
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8, loc="best")

    # theta_CLS panel: baseline solve + wrong.
    xs, acc, wrong = [], [], []
    for th in C.THETA_CLS_AXIS:
        entry = data.get(f"o4_cls_base_{_tag(th)}")
        if not entry or not entry["seeds"]:
            continue
        a = _mean(entry["seeds"], "accuracy")
        w = _mean(entry["seeds"], "wrong")
        if a is None:
            continue
        xs.append(th); acc.append(a); wrong.append(w if w is not None else float("nan"))
    if xs:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]; acc = [acc[i] for i in order]
        wrong = [wrong[i] for i in order]
        ax1.plot(xs, acc, "-o", color=C_MAIN, lw=2.0, label="solve rate")
        ax1b = ax1.twinx()
        ax1b.plot(xs, wrong, "--^", color=C_ALT, lw=1.4, alpha=0.7,
                  label="wrong-answer count")
        ax1b.set_ylabel("wrong-answer count")
        any_data = True
    else:
        skipped.append("o4:o4_cls_base")
    ax1.set_xlabel("θ_CLS (eval)")
    ax1.set_ylabel("solve rate")
    ax1.set_title("O4 — θ_CLS sensitivity (baseline)",
                  fontsize=11, fontweight="bold", color=INK)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, loc="best")

    if not any_data:
        plt.close(fig)
        skipped.append("o4:ALL(no data)")
        return
    fig.tight_layout()
    _save(fig, "o4_thresholds.pdf")


# --------------------------------------------------------------------------
# O3 — per-pass-index unsound compounding (deduce-to-fixpoint).
# --------------------------------------------------------------------------

def load_o3_per_pass() -> dict[str, list[tuple[int, float, int]]] | None:
    """Return {config: [(pass_index, unsound_rate, deduced), ...]} sorted by
    pass_index, or None if the per-pass table is absent/empty."""
    if not os.path.exists(O3_PER_PASS_CSV):
        return None
    by_cfg: dict[str, list[tuple[int, float, int]]] = {}
    with open(O3_PER_PASS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            rate = _fnum(r.get("unsound_rate"))
            ded = _fnum(r.get("deduced"))
            pi = _fnum(r.get("pass_index"))
            if rate is None or pi is None:
                continue
            by_cfg.setdefault(r["config"], []).append(
                (int(pi), rate, int(ded) if ded is not None else 0))
    for cfg_name in by_cfg:
        by_cfg[cfg_name].sort(key=lambda t: t[0])
    return by_cfg or None


# O3 per-pass config -> plot label + color.
O3_PP_STYLE = {
    "o3_d2": ("2 passes", C_MAIN),
    "o3_d4": ("4 passes", C_ALT),
    "o3_fix": ("fixpoint (cap 16)", STRAT_COLORS["late"]),
    "o3_d4_noaug": ("4 passes, no aug", STRAT_COLORS["early"]),
}


def fig_o3_per_pass() -> None:
    by_cfg = load_o3_per_pass()
    if not by_cfg:
        skipped.append("o3_per_pass:ALL(no data)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor("white")
    any_data = False
    for cfg_name, (label, color) in O3_PP_STYLE.items():
        series = by_cfg.get(cfg_name)
        if not series:
            skipped.append(f"o3_per_pass:{cfg_name}")
            continue
        xs = [t[0] for t in series]
        rate = [t[1] for t in series]
        ax.plot(xs, rate, "-o", color=color, lw=2.0, label=label)
        any_data = True
    if not any_data:
        plt.close(fig)
        skipped.append("o3_per_pass:ALL(no data)")
        return
    ax.set_xlabel("deduce pass index")
    ax.set_ylabel("unsound-elimination rate (per pass)")
    ax.set_title("O3 — unsound rate vs deduce pass (compounding)",
                 fontsize=12, fontweight="bold", color=INK)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "o3_per_pass_compounding.pdf")


def main() -> None:
    data = load_summary()
    fig_o1a(data)
    fig_o1b(data)
    fig_o2()
    fig_o4(data)
    fig_o3_per_pass()

    print("\n=== plot_all summary ===", flush=True)
    if written:
        print("Figures written:", flush=True)
        for p in written:
            print(f"  {p}", flush=True)
    else:
        print("No figures written (no data found).", flush=True)
    if skipped:
        print("Skipped for lack of data:", flush=True)
        for s in skipped:
            print(f"  {s}", flush=True)


if __name__ == "__main__":
    main()
