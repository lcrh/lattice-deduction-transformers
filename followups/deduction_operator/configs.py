"""E3 (deduction_operator) eval matrix — the single source of truth.

E3 is ENTIRELY eval-only. Every config re-evaluates a *frozen* E1 checkpoint
under a different inference operating point (loop count, deduction passes,
thresholds) — no training happens here. The E1 checkpoints are consumed
strictly by their fixed exchange-volume paths
(`/checkpoints/followups/e1/<input>_seed<N>.pt`) and E3's own eval artifacts
land under `/checkpoints/followups/e3/` with the

    <evalconfig>__on__<input>_seed<N>.eval.json (+ .eval.jsonl)

naming from `followups/README.md`. Nothing is retrained; the input checkpoints
are read-only.

Each run is a Modal eval-only entrypoint launched INDIVIDUALLY with:

    uv run modal run --detach experiments/sudoku/eval_only.py \
        --checkpoint /checkpoints/followups/e1/<input>_seed<N>.pt \
        <eval flags> \
        --ckpt-subdir followups/e3 \
        --ckpt-name <evalconfig>__on__<input>_seed<N> \
        --skip-if-done

The sub-studies (see README):
  O1 baseline loop-scaling : L_eval in {1,2,4,8,16,32,64,128} on `baseline`.
  O1 transfer matrix       : L_train in {1,2,4,8,16,32} (E1 d1_L*) x the same
                             L_eval axis; plus the d2_final_only row (expected
                             to BREAK — deep-sup vs final-only).
  O3 deduce-to-fixpoint    : deduce_passes 2 / 4 / 0(fixpoint) on `baseline`,
                             plus a deduce_passes=4 no-aug control.
  O4 threshold sensitivity : theta_elim sweep on `baseline` AND `d4_sym`;
                             theta_CLS sweep on `baseline`.

CLI:
    uv run python followups/deduction_operator/configs.py list           # all
    uv run python followups/deduction_operator/configs.py list o1_scale  # filter
    uv run python followups/deduction_operator/configs.py list o3
    uv run python followups/deduction_operator/configs.py status         # done N/M
    uv run python followups/deduction_operator/configs.py remaining      # missing only

Runs are STATEFUL / idempotent: every emitted launch command carries
`--skip-if-done`, so a whole-sweep re-launch executes only the (config, seed)
pairs whose `.eval.json` has not landed yet. `status` / `remaining` query the
Modal volume (degrading gracefully when it is unavailable).

This module is importable: `collect.py` and `plot_all.py` read `CONFIGS` for
the expected per-config eval flags (input checkpoint, eval_n_loops,
deduce_passes, threshold, cls_threshold, augment) and the sub-study grouping.
"""

from __future__ import annotations

import argparse
import sys

from followups import _common

# --------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------

EVAL_ENTRYPOINT = "experiments/sudoku/eval_only.py"
CKPT_SUBDIR = "followups/e3"          # where E3 eval artifacts land
E1_SUBDIR = "followups/e1"            # where the consumed input checkpoints live
VOLUME_NAME = _common.VOLUME_NAME

# The full L_eval axis reused by every loop-scaling study (O1).
L_EVAL_AXIS = (1, 2, 4, 8, 16, 32, 64, 128)
# The L_train axis for the transfer matrix — E1's d1_L* tied-loop checkpoints.
L_TRAIN_AXIS = (1, 2, 4, 8, 16, 32)
# Thresholds swept in O4.
THETA_ELIM_AXIS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
THETA_CLS_AXIS = (0.5, 0.55, 0.6, 0.7, 0.8)

# The 1,000-puzzle subsample used across E3 (matches E1's n_eval_puzzles=1000).
N_EVAL = 1000

# E1 native loop counts (documented for the eval_n_loops == native sanity gate
# and for collect.py to know each input's L_train). d1_L<L> trains at L loops;
# baseline / d2_final_only / d4_sym all train at 16 loops.
E1_NATIVE_LOOPS: dict[str, int] = {
    "baseline": 16,
    "d2_final_only": 16,
    "d4_sym": 16,
    **{f"d1_L{L}": L for L in L_TRAIN_AXIS},
}


# --------------------------------------------------------------------------
# Config data model.
#   name -> {
#     "study":   sub-study tag (grouping / headers),
#     "input":   E1 config name consumed (checkpoint path built from it),
#     "n_seeds": how many seeds of the INPUT checkpoint to evaluate,
#     "eval_flags": {eval_only.py flag: value} — only the deviations from the
#                   EVAL_DEFAULTS below; DROP removes a flag entirely,
#     "note":    optional human note.
#   }
# The output ckpt-name is always `<name>__on__<input>_seed<N>`.
# --------------------------------------------------------------------------

# Eval flags emitted on EVERY command (the shared eval protocol). A config's
# `eval_flags` only carries its one-factor deviation from these.
EVAL_DEFAULTS: dict[str, object] = {
    "n_eval": N_EVAL,
    # eval_only.py's own defaults for the operating point (the tuned point):
    "threshold": 0.10,        # theta_elim
    "cls_threshold": 0.6,     # theta_CLS (eval-time)
    "augment": True,          # eval-time aug ensembling ON (matches run.py eval)
    "eval_n_loops": 0,        # 0 = checkpoint native loops
    "deduce_passes": 1,       # 1 = legacy single pass
}

# Boolean flags emitted as a bare `--flag` when True, omitted when False.
BOOL_FLAGS = {"augment"}

# Sentinel: drop a flag entirely (emit neither the flag nor a value).
DROP = "__DROP__"

CONFIGS: dict[str, dict] = {}


def _add(name, study, input_cfg, eval_flags, n_seeds=1, note=""):
    CONFIGS[name] = {
        "study": study,
        "input": input_cfg,
        "n_seeds": n_seeds,
        "eval_flags": dict(eval_flags),
        "note": note,
    }


# --------------------------------------------------------------------------
# O1 — baseline loop-scaling. L_eval sweep on the E1 baseline (2K).
# 1 seed (exploratory scan) except L_eval=16 which is the SANITY GATE (== the
# baseline's native operating point) and gets 3 seeds to anchor the curve.
# --------------------------------------------------------------------------
for L in L_EVAL_AXIS:
    is_native = (L == 16)
    _add(
        f"o1_scale_L{L}", "O1-scale", "baseline",
        {"eval_n_loops": L},
        n_seeds=3 if is_native else 1,
        note=("SANITY GATE: L_eval=16 == baseline native n_loops -> must "
              "reproduce the no-flag baseline eval." if is_native
              else f"loop-scaling probe at L_eval={L}."),
    )

# --------------------------------------------------------------------------
# O1 — transfer matrix: L_train (E1 d1_L*) x L_eval. 48 cells, 1 seed each
# (exploratory heatmap). Consumes /checkpoints/followups/e1/d1_L<Ltrain>_seed0.pt.
# --------------------------------------------------------------------------
for Lt in L_TRAIN_AXIS:
    for Le in L_EVAL_AXIS:
        _add(
            f"o1_xfer_Lt{Lt}_Le{Le}", "O1-transfer", f"d1_L{Lt}",
            {"eval_n_loops": Le},
            n_seeds=1,
            note=(f"transfer: train L={Lt}, eval L={Le}."
                  + (" (eval_n_loops==native diagonal)" if Le == Lt else "")),
        )

# --------------------------------------------------------------------------
# O1 — d2_final_only row: the SAME L_eval sweep on the final-only-supervised
# checkpoint. Deep supervision trains every iteration to be a valid readout, so
# this row is expected to BREAK when unrolled off L_train — ties back to E1-D2.
# --------------------------------------------------------------------------
for Le in L_EVAL_AXIS:
    _add(
        f"o1_d2fo_Le{Le}", "O1-transfer-d2", "d2_final_only",
        {"eval_n_loops": Le},
        n_seeds=1,
        note=f"d2_final_only unrolled to L_eval={Le} (expected to degrade "
             "off the trained depth).",
    )

# --------------------------------------------------------------------------
# O3 — deduce-to-fixpoint before branching, on the baseline. 3 seeds (table
# rows). o3_d4_noaug is the augmentation-ensembling control (same passes, aug
# off) to separate iterated-deduction from the implicit test-time aug ensemble.
# --------------------------------------------------------------------------
_add(
    "o3_d2", "O3", "baseline",
    {"deduce_passes": 2},
    n_seeds=3,
    note="2 deduction passes per round before deciding.",
)
_add(
    "o3_d4", "O3", "baseline",
    {"deduce_passes": 4},
    n_seeds=3,
    note="4 deduction passes per round.",
)
_add(
    "o3_fix", "O3", "baseline",
    {"deduce_passes": 0, "deduce_pass_cap": 16},
    n_seeds=3,
    note="deduce to fixpoint (passes=0), safety cap 16.",
)
_add(
    "o3_d4_noaug", "O3", "baseline",
    {"deduce_passes": 4, "augment": False},
    n_seeds=3,
    note="4 passes with eval-time aug OFF — control isolating iterated "
         "deduction from the aug-ensembling effect.",
)

# --------------------------------------------------------------------------
# O4 — operating-point sensitivity.
#   theta_elim sweep on baseline AND d4_sym (symmetric-BCE checkpoint).
#   theta_CLS sweep on baseline.
# 1 seed each (exploratory sensitivity scans). The tuned points
# (theta_elim=0.1, theta_CLS=0.6) are the operating point already covered by
# the O1 baseline native gate, but re-emitted here so each sweep is a complete
# self-contained curve.
# --------------------------------------------------------------------------
def _fmt(x: float) -> str:
    """theta value -> compact tag (0.05 -> 005, 0.1 -> 01, 0.55 -> 055)."""
    return str(x).replace("0.", "0").replace(".", "")


for th in THETA_ELIM_AXIS:
    _add(
        f"o4_elim_base_{_fmt(th)}", "O4-elim-baseline", "baseline",
        {"threshold": th},
        n_seeds=1,
        note=f"theta_elim={th} on baseline.",
    )
for th in THETA_ELIM_AXIS:
    _add(
        f"o4_elim_sym_{_fmt(th)}", "O4-elim-d4sym", "d4_sym",
        {"threshold": th},
        n_seeds=1,
        note=f"theta_elim={th} on d4_sym (symmetric BCE).",
    )
for th in THETA_CLS_AXIS:
    _add(
        f"o4_cls_base_{_fmt(th)}", "O4-cls-baseline", "baseline",
        {"cls_threshold": th},
        n_seeds=1,
        note=f"theta_CLS={th} on baseline.",
    )


# --------------------------------------------------------------------------
# Ordered sub-study grouping for `list` output (comment headers).
# --------------------------------------------------------------------------
STUDY_ORDER = [
    "O1-scale", "O1-transfer", "O1-transfer-d2",
    "O3", "O4-elim-baseline", "O4-elim-d4sym", "O4-cls-baseline",
]
STUDY_LABEL = {
    "O1-scale": "O1  baseline loop-scaling (L_eval sweep; L=16 is the sanity gate)",
    "O1-transfer": "O1  train/eval loop-transfer matrix (L_train x L_eval)",
    "O1-transfer-d2": "O1  d2_final_only transfer row (expected to break)",
    "O3": "O3  deduce-to-fixpoint before branching (baseline; +no-aug control)",
    "O4-elim-baseline": "O4  theta_elim sweep on baseline",
    "O4-elim-d4sym": "O4  theta_elim sweep on d4_sym (symmetric BCE)",
    "O4-cls-baseline": "O4  theta_CLS sweep on baseline",
}


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

_flag = _common.flag

# Stable emission order for eval flags (readability only).
FLAG_ORDER = (
    "n_eval", "eval_n_loops", "deduce_passes", "deduce_pass_cap",
    "threshold", "cls_threshold", "augment",
)


def input_checkpoint(config_name: str, seed: int) -> str:
    """Fixed E1 volume path of the input checkpoint this config consumes."""
    inp = CONFIGS[config_name]["input"]
    return f"/checkpoints/{E1_SUBDIR}/{inp}_seed{seed}.pt"


def output_name(config_name: str, seed: int) -> str:
    """E3 output ckpt-name: `<evalconfig>__on__<input>_seed<N>`."""
    inp = CONFIGS[config_name]["input"]
    return f"{config_name}__on__{inp}_seed{seed}"


def effective_flags(config_name: str) -> dict[str, object]:
    """Effective eval-flag dict (EVAL_DEFAULTS + per-config deviations).

    DROP-valued flags are removed. Used by collect.py to know the expected
    operating point (eval_n_loops, deduce_passes, threshold, cls_threshold,
    augment) for each config.
    """
    cfg = CONFIGS[config_name]
    eff = dict(EVAL_DEFAULTS)
    eff.update(cfg["eval_flags"])
    return {k: v for k, v in eff.items() if v is not DROP}


def effective_n_loops(config_name: str) -> int:
    """Loops actually used at eval: eval_n_loops if >0 else input's native L."""
    eff = effective_flags(config_name)
    enl = int(eff.get("eval_n_loops", 0) or 0)
    if enl > 0:
        return enl
    return E1_NATIVE_LOOPS[CONFIGS[config_name]["input"]]


def launch_command(config_name: str, seed: int) -> str:
    """Build the eval-only `uv run modal run --detach ...` command."""
    eff = effective_flags(config_name)
    ckpt = input_checkpoint(config_name, seed)
    out = output_name(config_name, seed)
    parts = [f"uv run modal run --detach {EVAL_ENTRYPOINT}"]
    parts.append(f"{_flag('checkpoint')} {ckpt}")
    # Emit eval flags in a stable order (present keys only).
    ordered = list(FLAG_ORDER) + [k for k in eff if k not in FLAG_ORDER]
    emitted: set[str] = set()
    for name in ordered:
        if name in emitted or name not in eff:
            continue
        emitted.add(name)
        val = eff[name]
        if name in BOOL_FLAGS:
            # eval_only.py's `augment` defaults True; only emit the negative
            # form when explicitly False. (Modal maps --no-augment -> False.)
            if val:
                # default already True; emitting --augment is harmless but
                # noisy — emit only when it differs from the eval_only default.
                if EVAL_DEFAULTS.get(name) is not True:
                    parts.append(_flag(name))
            else:
                parts.append(f"--no-{name.replace('_', '-')}")
        else:
            parts.append(f"{_flag(name)} {val}")
    # Deterministic output routing under followups/e3.
    parts.append(f"{_flag('ckpt_subdir')} {CKPT_SUBDIR}")
    parts.append(f"{_flag('ckpt_name')} {out}")
    parts.append("--skip-if-done")
    return " \\\n    ".join(parts)


def iter_runs(prefix: str = ""):
    """Yield (config_name, seed) for every run, optionally filtered by prefix.

    Filter matches config name (substring) or study tag.
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


# --------------------------------------------------------------------------
# Volume querying (status / remaining). Degrades gracefully — never aborts on
# status; matches E1/E4's non-aborting status behavior.
# --------------------------------------------------------------------------

def _try_volume_done_set() -> tuple[set[str] | None, str | None]:
    """(done-set, None) on success; (None, error) on failure.

    Delegates to `_common.volume_done_set` on the E3 subdir. The done-set
    holds `<evalconfig>__on__<input>_seed<N>` basenames with a landed
    `.eval.json`.
    """
    try:
        return _common.volume_done_set(CKPT_SUBDIR), None
    except Exception as e:  # noqa: BLE001 — modal missing / unauthed / dir absent
        return None, f"{type(e).__name__}: {e}"


def _matches(name: str, study: str, prefix: str) -> bool:
    return (not prefix) or (prefix.lower() in name.lower()
                            or prefix.lower() in study.lower())


# --------------------------------------------------------------------------
# Subcommands.
# --------------------------------------------------------------------------

def _print_list(prefix: str = "") -> None:
    print("# " + "=" * 72, flush=True)
    print("# E3 (deduction_operator) EVAL-ONLY launch commands.", flush=True)
    print("#", flush=True)
    print("# PREREQUISITE: the consumed E1 checkpoints must already be on the", flush=True)
    print("#   volume at /checkpoints/followups/e1/<input>_seed<N>.pt", flush=True)
    print("#   (baseline, d1_L*, d2_final_only, d4_sym). E3 retrains NOTHING.", flush=True)
    print("#", flush=True)
    print("# 1. Run the SANITY GATE FIRST: o1_scale_L16__on__baseline_seed0", flush=True)
    print("#    (--eval-n-loops 16 on the 16-loop baseline) must reproduce the", flush=True)
    print("#    no-flag baseline eval before any scaling result is interpreted.", flush=True)
    print("# 2. Launch each command INDIVIDUALLY (one `modal run --detach` per", flush=True)
    print("#    run). Do NOT wrap these in a shell loop.", flush=True)
    print("# 3. Eval artifacts land at", flush=True)
    print("#    /checkpoints/followups/e3/<evalconfig>__on__<input>_seed<N>.eval.json", flush=True)
    print("# 4. Every command carries --skip-if-done: re-running the sweep only", flush=True)
    print("#    executes (config, seed) pairs whose eval.json has not landed.", flush=True)
    if prefix:
        print(f"#\n# FILTER: only configs/studies matching {prefix!r}.", flush=True)
    print("# " + "=" * 72, flush=True)

    n_cmds = 0
    for study in STUDY_ORDER:
        names = [n for n, c in CONFIGS.items()
                 if c["study"] == study and _matches(n, study, prefix)]
        if not names:
            continue
        print(f"\n# --- {STUDY_LABEL[study]} ---", flush=True)
        for name in names:
            cfg = CONFIGS[name]
            note = f"  # {cfg['note']}" if cfg["note"] else ""
            print(f"# {name}  (input={cfg['input']}, {cfg['n_seeds']} seed(s))"
                  f"{note}", flush=True)
            for seed in range(cfg["n_seeds"]):
                print(launch_command(name, seed), flush=True)
                print("", flush=True)
                n_cmds += 1
    print(f"# Total: {n_cmds} launch commands"
          + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def _print_status(prefix: str = "") -> None:
    done, err = _try_volume_done_set()
    if err is not None:
        print(f"# [status] Modal volume {VOLUME_NAME!r} unavailable ({err}); "
              "counts below are EXPECTED totals (0 landed).", flush=True)
        done = set()
    tag = "  (volume unavailable — expected totals)" if err else ""
    print(f"# E3 volume status  (/{CKPT_SUBDIR}/, {len(done)} eval.json files)"
          + tag, flush=True)
    n_total = n_done = 0
    for study in STUDY_ORDER:
        study_names = [n for n, c in CONFIGS.items()
                       if c["study"] == study and _matches(n, study, prefix)]
        if not study_names:
            continue
        print(f"\n# {STUDY_LABEL[study]}", flush=True)
        for name in study_names:
            cfg = CONFIGS[name]
            got = sum(1 for s in range(cfg["n_seeds"])
                      if output_name(name, s) in done)
            n_total += cfg["n_seeds"]
            n_done += got
            mark = "OK " if got == cfg["n_seeds"] else "   "
            print(f"  {mark}{name}: {got}/{cfg['n_seeds']} done", flush=True)
    print(f"\n# Overall: {n_done}/{n_total} runs done"
          + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def _print_remaining(prefix: str = "") -> None:
    done, err = _try_volume_done_set()
    if err is not None:
        print(f"# [remaining] could not query Modal volume {VOLUME_NAME!r}: "
              f"{err}", flush=True)
        print("# [remaining] (need modal auth + network). Aborting.", flush=True)
        sys.exit(1)
    print("# " + "=" * 72, flush=True)
    print("# E3 REMAINING eval launch commands (missing eval.json only).", flush=True)
    print("# Launch each INDIVIDUALLY (no shell loop). --skip-if-done is a", flush=True)
    print("# belt-and-suspenders no-op if one lands between query and launch.", flush=True)
    print("# " + "=" * 72, flush=True)
    n_cmds = 0
    for study in STUDY_ORDER:
        study_names = [n for n, c in CONFIGS.items()
                       if c["study"] == study and _matches(n, study, prefix)]
        if not study_names:
            continue
        header_done = False
        for name in study_names:
            cfg = CONFIGS[name]
            missing = [s for s in range(cfg["n_seeds"])
                       if output_name(name, s) not in done]
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
    if n_cmds == 0:
        print("\n# Nothing remaining — all runs have a landed eval.json.",
              flush=True)
    else:
        print(f"# Total remaining: {n_cmds} launch commands"
              + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    p_list = sub.add_parser("list", help="print launch commands (one per run)")
    p_list.add_argument("filter", nargs="?", default="",
                        help="optional prefix/substring filter "
                             "(e.g. o1_scale, o1_xfer, o3, o4_elim)")
    p_status = sub.add_parser("status", help="query volume: N/M seeds done per config")
    p_status.add_argument("filter", nargs="?", default="")
    p_remaining = sub.add_parser("remaining",
                                 help="launch commands for missing runs only")
    p_remaining.add_argument("filter", nargs="?", default="")
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
