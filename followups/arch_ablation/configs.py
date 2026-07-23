"""E1 (arch_ablation) run matrix — the single source of truth.

Every config in the README's D1 (C1/C2/C3/C4), D2, D3, D4 tables is encoded
here as data: config-name -> {flag overrides} + n_seeds. Baselines are RE-RUN
under the same eval protocol (not quoted), so `baseline` appears in the matrix.

Each run is a Modal entrypoint launched with:

    uv run modal run --detach experiments/sudoku/run.py -- <flags>

Checkpoints land at the deterministic exchange path
`/checkpoints/followups/e1/<config>_seed<N>.pt` (+ .eval.json / .eval.jsonl /
.train_curve.jsonl) via `--ckpt-subdir followups/e1 --ckpt-name <config>_seed<N>`.
The C1 checkpoints (`d1_L<L>`) double as E3's `L_train` axis.

CLI:
    uv run python followups/arch_ablation/configs.py list          # all configs
    uv run python followups/arch_ablation/configs.py list d1_L     # filter prefix
    uv run python followups/arch_ablation/configs.py list d4
    uv run python followups/arch_ablation/configs.py status        # done N/M per config
    uv run python followups/arch_ablation/configs.py remaining     # cmds for missing only

Runs are STATEFUL / idempotent: every emitted launch command carries
`--skip-if-done`, so a whole-sweep re-launch executes only the (config, seed)
pairs whose `.eval.json` has not landed yet. `status` / `remaining` query the
Modal volume to report / target the missing pairs.

This module is importable: `collect.py` and `plot_all.py` read `CONFIGS` for
the expected per-config flag overrides (n_loops / num_layers / dim / steps /
supervise / softmax_loss_weight / bce_* / conflict_loss_weight, ...).
"""

from __future__ import annotations

import argparse
import sys

from followups import _common

# --------------------------------------------------------------------------
# Baseline (all configs are a one-factor deviation from this).
#   4 layers x 16 loops, dim 128, bce 4.0/0.5, softmax_loss_weight 0.2,
#   conflict_loss_weight 0.1, supervise "all", 2000 steps, 1K train / 1K eval.
# These match experiments/sudoku/run.py defaults EXCEPT steps/n_*_puzzles,
# which the ablation baseline pins explicitly. We therefore emit the ablation
# baseline defaults on every launch command so a config's `overrides` dict
# only needs to carry its one-factor deviation.
# --------------------------------------------------------------------------

# Ablation-wide defaults, applied to every run unless a config overrides them.
BASE_DEFAULTS: dict[str, object] = {
    "steps": 2000,
    "n_train_puzzles": 1000,
    "n_eval_puzzles": 1000,
    "num_layers": 4,
    "dim": 128,
    "n_loops": 16,
    "supervise": "all",
    "softmax_loss_weight": 0.2,
    "conflict_loss_weight": 0.1,
    "bce_pos_mult": 4.0,
    "bce_neg_mult": 0.5,
    "eval_every": 100,
}

# Sentinel: an override value of DROP means "do not emit this flag at all",
# used to unset --n-train-puzzles for the full-split ("bigdata"/"max") rows.
DROP = "__DROP__"

CKPT_SUBDIR = "followups/e1"


# --------------------------------------------------------------------------
# The run matrix. Each entry:
#   name -> {"study": <sub-study tag>, "n_seeds": int, "overrides": {...},
#            "note": <optional str>}
# `overrides` are ONLY the deviations from BASE_DEFAULTS. `collect.py` reads
# the effective params via `effective_flags(name)`.
# --------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {}


def _add(name, study, overrides, n_seeds=3, note=""):
    CONFIGS[name] = {
        "study": study,
        "n_seeds": n_seeds,
        "overrides": overrides,
        "note": note,
    }


# --- Baseline (re-run control; shared by D2/D3/D4) -------------------------
_add(
    "baseline", "baseline",
    {},  # pure BASE_DEFAULTS
    n_seeds=3,
    note="4x16 dim128, bce 4.0/0.5 (8x), lambda_ce 0.2, deep-sup control.",
)

# --- D1-C1: tied loop sweep (params fixed ~800K; per-forward FLOPs prop L) --
# L = n_loops in {1,2,4,8,16,32}, 4 layers dim128, steps 2000, 3 seeds.
# d1_L16 == baseline architecture, but kept under the C1 name so E3 can read
# the full L_train axis from /checkpoints/followups/e1/d1_L<L>_seed<N>.pt.
for L in (1, 2, 4, 8, 16, 32):
    _add(
        f"d1_L{L}", "D1-C1",
        {"n_loops": L},
        n_seeds=3,
        note=("non-recurrent null hypothesis (L=1)" if L == 1
              else ("== baseline architecture (kept as C1 name for E3 L_train axis)"
                    if L == 16 else "")),
    )

# --- D1-C2: escalation ladder (can data/compute buy back recursion?) -------
# eval_every set to steps/100 so each run yields ~100 curve points.
_add(
    "d1_L2_cm", "D1-C2",
    {"n_loops": 2, "steps": 16000, "n_train_puzzles": 1000, "eval_every": 160},
    n_seeds=3,
    note="L=2 parity point, 1x baseline training FLOPs.",
)
_add(
    "d1_L1_cm", "D1-C2",
    {"n_loops": 1, "steps": 32000, "n_train_puzzles": 1000, "eval_every": 320},
    n_seeds=3,
    note="L=1, 32K steps = 1x baseline training FLOPs (parity).",
)
_add(
    "d1_L1_cm4x", "D1-C2",
    {"n_loops": 1, "steps": 128000, "n_train_puzzles": 1000, "eval_every": 1280},
    n_seeds=2,
    note="L=1, 128K steps = 4x compute. ~30 B200-min. 2 seeds.",
)
_add(
    "d1_L1_bigdata", "D1-C2",
    {"n_loops": 1, "steps": 128000, "n_train_puzzles": DROP, "eval_every": 1280},
    n_seeds=2,
    note="L=1, 128K steps, FULL train split (data bottleneck removed). 2 seeds.",
)

# --- D1-C3: param-matched width-vs-depth ladder (non-recurrent shapes) -----
# All L=1, steps 2000, dims picked for n_heads=4 divisibility. 3 seeds
# (README allows dropping to 2 under budget pressure; kept at 3).
# d1_L1 (4x128) is shared with C1 — not re-declared here.
_add(
    "d1_shape8x92", "D1-C3",
    {"n_loops": 1, "num_layers": 8, "dim": 92},
    n_seeds=3,
    note="~810K params, L=1. Budget option: drop to 2 seeds.",
)
_add(
    "d1_shape16x64", "D1-C3",
    {"n_loops": 1, "num_layers": 16, "dim": 64},
    n_seeds=3,
    note="~790K params, L=1. Budget option: drop to 2 seeds.",
)
_add(
    "d1_shape32x44", "D1-C3",
    {"n_loops": 1, "num_layers": 32, "dim": 44},
    n_seeds=3,
    note="~740K params, L=1. Budget option: drop or 2 seeds.",
)

# --- D1-C4: params escalation (generous untied controls; FLOPs-parity steps)
_add(
    "d1_untied8", "D1-C4",
    {"n_loops": 1, "num_layers": 8, "dim": 128, "steps": 16000},
    n_seeds=3,
    note="~1.6M params, 1/8 fwd FLOPs -> 16K steps for FLOPs parity.",
)
_add(
    "d1_untied16", "D1-C4",
    {"n_loops": 1, "num_layers": 16, "dim": 128, "steps": 8000},
    n_seeds=3,
    note="~3.2M params, 1/4 fwd FLOPs -> 8K steps. Matches tied d1_L4 layer-apps.",
)
_add(
    "d1_wide", "D1-C4",
    {"n_loops": 1, "num_layers": 4, "dim": 256, "steps": 8000},
    n_seeds=3,
    note="~3.2M params (width), 1/4 fwd FLOPs -> 8K steps. Depth-vs-width vs d1_untied16.",
)
_add(
    "d1_untied16_max", "D1-C4",
    {"n_loops": 1, "num_layers": 16, "dim": 128, "steps": 32000,
     "n_train_puzzles": DROP},
    n_seeds=2,
    note="No-excuses challenger: 4x params, 4x compute, full split. ~30 B200-min. 2 seeds.",
)

# --- D2: deep supervision --------------------------------------------------
# baseline is the deep-sup (supervise all) control, already in the matrix.
_add(
    "d2_final_only", "D2",
    {"supervise": "final", "eval_every": 50},
    n_seeds=3,
    note="Supervise final iteration only; eval_every 50 for curve resolution.",
)

# --- D3: does L_CE speed up learning? --------------------------------------
# baseline is lambda_ce=0.2 control.
_add(
    "d3_ce0", "D3",
    {"softmax_loss_weight": 0.0, "eval_every": 50},
    n_seeds=3,
    note="lambda_ce=0. eval_every 50. Check calls/solve + unsound to attribute.",
)
_add(
    "d3_ce1", "D3",
    {"softmax_loss_weight": 1.0, "eval_every": 50},
    n_seeds=3,
    note="OPTIONAL (budget: drop to 2 seeds or skip). lambda_ce=1.0. eval_every 50.",
)

# --- D4: soundness-pressure knobs (never trim) -----------------------------
# baseline is the 8x control (bce 4.0/0.5).
_add(
    "d4_sym", "D4",
    {"bce_pos_mult": 0.5, "bce_neg_mult": 0.5},
    n_seeds=3,
    note="Symmetric BCE (ratio 1x). CLS head on.",
)
_add(
    "d4_ratio2", "D4",
    {"bce_pos_mult": 1.0, "bce_neg_mult": 0.5},
    n_seeds=3,
    note="Ratio 2x. CLS head on.",
)
_add(
    "d4_ratio32", "D4",
    {"bce_pos_mult": 16.0, "bce_neg_mult": 0.5},
    n_seeds=3,
    note="Ratio 32x. CLS head on.",
)
_add(
    "d4_nocls", "D4",
    {"bce_pos_mult": 4.0, "bce_neg_mult": 0.5, "conflict_loss_weight": 0.0},
    n_seeds=3,
    note="8x BCE but NO CLS head (conflict_loss_weight 0 -> cls_token off; "
         "conflicts via empty-cell test only).",
)


# Ordered sub-study grouping for `list` output (comment headers).
STUDY_ORDER = [
    "baseline", "D1-C1", "D1-C2", "D1-C3", "D1-C4", "D2", "D3", "D4",
]
STUDY_LABEL = {
    "baseline": "BASELINE (re-run control; sanity gate)",
    "D1-C1": "D1-C1  tied loop sweep (params fixed; per-forward FLOPs prop L)",
    "D1-C2": "D1-C2  escalation ladder (data/compute buy-back)",
    "D1-C3": "D1-C3  param-matched width-vs-depth shapes (L=1)",
    "D1-C4": "D1-C4  params escalation (untied controls; FLOPs-parity steps)",
    "D2": "D2  deep supervision",
    "D3": "D3  L_CE learning-speed",
    "D4": "D4  soundness-pressure knobs",
}


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

# Stable emission order for launch-command flags (readability only).
FLAG_ORDER = (
    "steps", "n_train_puzzles", "n_eval_puzzles",
    "num_layers", "dim", "n_loops", "supervise",
    "softmax_loss_weight", "conflict_loss_weight",
    "bce_pos_mult", "bce_neg_mult", "eval_every",
)

# Re-exported for collect.py / plot_all.py, which key off the volume name.
VOLUME_NAME = _common.VOLUME_NAME


def effective_flags(config_name: str) -> dict[str, object]:
    """Return the effective flag dict for a config (BASE_DEFAULTS + overrides).

    DROP-valued overrides are removed entirely (flag not emitted). Used by
    collect.py to know the expected per-config params (n_loops, num_layers,
    dim, steps, ...).
    """
    cfg = CONFIGS[config_name]
    eff = dict(BASE_DEFAULTS)
    eff.update(cfg["overrides"])
    return {k: v for k, v in eff.items() if v is not DROP}


def launch_command(config_name: str, seed: int) -> str:
    """Build the `uv run modal run --detach ...` command for (config, seed)."""
    return _common.launch_command(
        config_name, seed, effective_flags(config_name),
        ckpt_subdir=CKPT_SUBDIR, flag_order=FLAG_ORDER, skip_if_done=True,
    )


def _volume_done_set() -> set[str]:
    """Set of `<config>_seed<N>` names with a landed eval.json (via _common)."""
    return _common.volume_done_set(CKPT_SUBDIR)


def iter_runs(prefix: str = ""):
    """Yield (config_name, seed) for every run, optionally filtered by prefix.

    Filter matches either the config name (substring) or the study tag
    (e.g. "d1_L", "d4", "D1-C2").
    """
    for study in STUDY_ORDER:
        for name, cfg in CONFIGS.items():
            if cfg["study"] != study:
                continue
            if prefix and not (prefix.lower() in name.lower()
                               or prefix.lower() in study.lower()):
                continue
            for seed in range(cfg["n_seeds"]):
                yield name, seed


def _print_list(prefix: str = "") -> None:
    print("# " + "=" * 72, flush=True)
    print("# E1 (arch_ablation) launch commands.", flush=True)
    print("#", flush=True)
    print("# 1. Run the SANITY-GATE baseline FIRST — it must reproduce before", flush=True)
    print("#    anything else launches (baseline_seed0 below).", flush=True)
    print("# 2. Launch each command INDIVIDUALLY (one `modal run --detach` per", flush=True)
    print("#    run). Do NOT wrap these in a shell loop.", flush=True)
    print("# 3. Checkpoints land at /checkpoints/followups/e1/<config>_seed<N>.pt", flush=True)
    if prefix:
        print(f"#\n# FILTER: only configs/studies matching {prefix!r}.", flush=True)
    print("# " + "=" * 72, flush=True)

    n_cmds = 0
    for study in STUDY_ORDER:
        names = [n for n, c in CONFIGS.items() if c["study"] == study
                 and (not prefix or prefix.lower() in n.lower()
                      or prefix.lower() in study.lower())]
        if not names:
            continue
        print(f"\n# --- {STUDY_LABEL[study]} ---", flush=True)
        for name in names:
            cfg = CONFIGS[name]
            note = f"  # {cfg['note']}" if cfg["note"] else ""
            print(f"# {name}  ({cfg['n_seeds']} seeds){note}", flush=True)
            for seed in range(cfg["n_seeds"]):
                print(launch_command(name, seed), flush=True)
                print("", flush=True)
                n_cmds += 1

    print(f"# Total: {n_cmds} launch commands"
          + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def _print_status(prefix: str = "") -> None:
    """Query the volume and print, per config, how many seeds have landed."""
    try:
        done = _volume_done_set()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, don't crash
        print(f"[status] could not query Modal volume {VOLUME_NAME!r}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        print("[status] (need modal auth + network). Aborting status.", flush=True)
        sys.exit(1)

    print(f"# E1 volume status  (/{CKPT_SUBDIR}/, {len(done)} eval.json files)",
          flush=True)
    n_total = n_done = 0
    for study in STUDY_ORDER:
        study_names = [n for n, c in CONFIGS.items() if c["study"] == study
                       and (not prefix or prefix.lower() in n.lower()
                            or prefix.lower() in study.lower())]
        if not study_names:
            continue
        print(f"\n# {STUDY_LABEL[study]}", flush=True)
        for name in study_names:
            cfg = CONFIGS[name]
            got = sum(1 for s in range(cfg["n_seeds"])
                      if f"{name}_seed{s}" in done)
            n_total += cfg["n_seeds"]
            n_done += got
            mark = "OK " if got == cfg["n_seeds"] else "   "
            print(f"  {mark}{name}: {got}/{cfg['n_seeds']} done", flush=True)
    print(f"\n# Overall: {n_done}/{n_total} runs done"
          + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def _print_remaining(prefix: str = "") -> None:
    """Print launch commands ONLY for (config, seed) pairs missing eval.json."""
    try:
        done = _volume_done_set()
    except Exception as exc:  # noqa: BLE001
        print(f"[remaining] could not query Modal volume {VOLUME_NAME!r}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        print("[remaining] (need modal auth + network). Aborting.", flush=True)
        sys.exit(1)

    print("# " + "=" * 72, flush=True)
    print("# E1 REMAINING launch commands (missing eval.json only).", flush=True)
    print("# Launch each INDIVIDUALLY (no shell loop). --skip-if-done is a", flush=True)
    print("# belt-and-suspenders no-op if one lands between query and launch.", flush=True)
    print("# " + "=" * 72, flush=True)
    n_cmds = 0
    for study in STUDY_ORDER:
        study_names = [n for n, c in CONFIGS.items() if c["study"] == study
                       and (not prefix or prefix.lower() in n.lower()
                            or prefix.lower() in study.lower())]
        if not study_names:
            continue
        header_done = False
        for name in study_names:
            cfg = CONFIGS[name]
            missing = [s for s in range(cfg["n_seeds"])
                       if f"{name}_seed{s}" not in done]
            if not missing:
                continue
            if not header_done:
                print(f"\n# --- {STUDY_LABEL[study]} ---", flush=True)
                header_done = True
            note = f"  # {cfg['note']}" if cfg["note"] else ""
            print(f"# {name}  (missing seeds: {missing}){note}", flush=True)
            for seed in missing:
                print(launch_command(name, seed), flush=True)
                print("", flush=True)
                n_cmds += 1
    print(f"# Total remaining: {n_cmds} launch commands"
          + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    p_list = sub.add_parser("list", help="print launch commands (one per run)")
    p_list.add_argument("filter", nargs="?", default="",
                        help="optional prefix/substring filter (e.g. d1_L, d4, D1-C2)")
    p_status = sub.add_parser("status", help="query volume: N/M seeds done per config")
    p_status.add_argument("filter", nargs="?", default="",
                          help="optional prefix/substring filter")
    p_remaining = sub.add_parser("remaining",
                                 help="launch commands for missing runs only")
    p_remaining.add_argument("filter", nargs="?", default="",
                             help="optional prefix/substring filter")
    args = parser.parse_args()

    if args.cmd == "list":
        _print_list(args.filter)
    elif args.cmd == "status":
        _print_status(args.filter)
    elif args.cmd == "remaining":
        _print_remaining(args.filter)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
