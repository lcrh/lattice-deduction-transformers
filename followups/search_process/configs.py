"""E2 (search_process) run matrix — the single source of truth.

E2 ablates the DECIDE and BACKTRACK halves of LDT's DPLL search loop. It has
BOTH a small set of TRAINING configs it owns (the weak 1K-step checkpoint and a
4K checkpoint for the model-strength figure, plus the S2 matched-policy trainer)
AND a large set of EVAL-ONLY configs that re-evaluate frozen checkpoints under
different decision / backtracking policies.

Two run kinds, encoded in each config's `kind`:

  kind="train"  -> `experiments/sudoku/run.py`, writes
                   /checkpoints/followups/e2/<config>_seed<N>.pt
                   (via --ckpt-subdir followups/e2 --ckpt-name <config>_seed<N>).
  kind="eval"   -> `experiments/sudoku/eval_only.py`, re-evaluates a FROZEN
                   checkpoint (referenced by its fixed volume path) under a new
                   operating point. Output lands at
                   /checkpoints/followups/e2/<config>__on__<input>_seed<N>.eval.json
                   (the `__on__` naming from followups/README.md).

Consumed inputs (read-only):
  * E1 baseline (2K steps):  /checkpoints/followups/e1/baseline_seed<N>.pt
  * E2 base_1k  (1K steps):  /checkpoints/followups/e2/base_1k_seed<N>.pt  (owned)
  * E2 base_4k  (4K steps):  /checkpoints/followups/e2/base_4k_seed<N>.pt  (owned)

Sub-studies (see README):
  S1  decision-policy scan (~10 selected combos) x {baseline, base_1k}, eval.
  S2  matched-vs-mismatched 2x2: train-P0 (=baseline / base_1k) & train-P*
      x eval-P0 & eval-P*.
  S3  backtracking policy list, eval on baseline + base_1k (phase 1) and the
      matched-training trainer (phase 2).
  S4  2-3 best (policy, backtrack) combos + baseline x {1K, 2K, 4K} checkpoints.

CLI:
    uv run python followups/search_process/configs.py list          # all
    uv run python followups/search_process/configs.py list s1        # filter
    uv run python followups/search_process/configs.py status         # done N/M
    uv run python followups/search_process/configs.py remaining      # missing only

Runs are STATEFUL / idempotent: every emitted launch command carries
`--skip-if-done`, so a whole-sweep re-launch executes only the (config, seed)
pairs whose `.eval.json` has not landed. `status`/`remaining` query the Modal
volume (degrading gracefully when unavailable — matches E1/E3 status behavior).

This module is importable: collect.py / plot_all.py read `CONFIGS` for the
expected per-config flags and the sub-study grouping.
"""

from __future__ import annotations

import argparse
import sys

from followups import _common

# --------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------

RUN_ENTRYPOINT = "experiments/sudoku/run.py"
EVAL_ENTRYPOINT = "experiments/sudoku/eval_only.py"
CKPT_SUBDIR = "followups/e2"        # where E2 checkpoints + eval artifacts land
E1_SUBDIR = "followups/e1"          # E1 baseline (consumed by S1/S3/S4)
VOLUME_NAME = _common.VOLUME_NAME

# 1,000-puzzle eval subsample used across E2 (matches E1/E3 n_eval).
N_EVAL = 1000

DROP = "__DROP__"

# --------------------------------------------------------------------------
# Training-config defaults (kind="train"). These mirror the E1 ablation
# baseline EXCEPT `steps`, which each training config pins. base_1k / base_4k
# train the weak / strong checkpoints this experiment owns; train_pstar is the
# S2 matched-policy run (trains with P*).
# --------------------------------------------------------------------------
TRAIN_DEFAULTS: dict[str, object] = {
    "steps": 2000,
    "n_train_puzzles": 1000,
    "n_eval_puzzles": N_EVAL,
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
TRAIN_FLAG_ORDER = (
    "steps", "n_train_puzzles", "n_eval_puzzles",
    "num_layers", "dim", "n_loops", "supervise",
    "softmax_loss_weight", "conflict_loss_weight",
    "bce_pos_mult", "bce_neg_mult", "eval_every",
    "cell_policy", "digit_policy", "backtrack", "geometric_p",
)

# --------------------------------------------------------------------------
# Eval-config defaults (kind="eval"). A config's eval_flags carries only its
# one-factor deviation. Matches eval_only.py's own defaults at the tuned point.
# --------------------------------------------------------------------------
EVAL_DEFAULTS: dict[str, object] = {
    "n_eval": N_EVAL,
    "threshold": 0.10,
    "cls_threshold": 0.6,
    "augment": True,
    "estimate_sequential": True,   # E2 key-cost metric — always report seq cost
    "cell_policy": "uniform",
    "digit_policy": "softmax",
    "backtrack": "root",
}
EVAL_FLAG_ORDER = (
    "n_eval", "threshold", "cls_threshold",
    "cell_policy", "digit_policy",
    "backtrack", "geometric_p", "learn_negation",
    "estimate_sequential", "augment",
)
# Bool flags emitted specially (bare --flag / --no-flag).
BOOL_FLAGS = {"augment", "estimate_sequential", "learn_negation"}


# --------------------------------------------------------------------------
# Config data model.
#   name -> {
#     kind:     "train" | "eval"
#     study:    sub-study tag (grouping / headers)
#     n_seeds:  int
#     note:     optional str
#     -- train-only --
#     overrides:  {run.py flag: value}  (deviations from TRAIN_DEFAULTS)
#     -- eval-only --
#     input:      base config name consumed; its checkpoint path is built via
#                 input_checkpoint(); "baseline" lives under E1, others under E2.
#     eval_flags: {eval_only.py flag: value} (deviations from EVAL_DEFAULTS)
#   }
# --------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {}

# Which subdir an input checkpoint lives under (baseline = E1; everything else
# that E2 trains lives under E2).
INPUT_SUBDIR = {
    "baseline": E1_SUBDIR,
    "base_1k": CKPT_SUBDIR,
    "base_4k": CKPT_SUBDIR,
    # S2 matched-policy trainers (trained below):
    "train_pstar_1k": CKPT_SUBDIR,
}


def _add_train(name, study, overrides, n_seeds=3, note=""):
    CONFIGS[name] = {
        "kind": "train", "study": study, "n_seeds": n_seeds,
        "overrides": dict(overrides), "note": note,
    }


def _add_eval(name, study, input_cfg, eval_flags, n_seeds=1, note=""):
    CONFIGS[name] = {
        "kind": "eval", "study": study, "input": input_cfg,
        "n_seeds": n_seeds, "eval_flags": dict(eval_flags), "note": note,
    }


# --------------------------------------------------------------------------
# TRAINING configs E2 owns.
# --------------------------------------------------------------------------
# base_1k: weak-deduction checkpoint (1K steps) — policy effects should be
# LARGE here. Consumed by S1/S3/S4 and the S4 model-strength figure.
_add_train(
    "base_1k", "TRAIN", {"steps": 1000, "eval_every": 50}, n_seeds=3,
    note="Weak 1K-step checkpoint E2 owns (lots of search). ~7 B200-min.",
)
# base_4k: strong checkpoint for the S4 budget axis (1K/2K/4K). 2K already
# exists as the E1 baseline; 4K trained here.
_add_train(
    "base_4k", "TRAIN", {"steps": 4000}, n_seeds=3,
    note="Strong 4K-step checkpoint for the S4 budget axis. ~15 B200-min.",
)
# --------------------------------------------------------------------------
# P0 / P* — the two S2 policies, defined ONCE here (single source of truth used
# by both the train_pstar_1k trainer and the S2 2x2 eval cells below).
#
# P* is a PLACEHOLDER pending S1 results. It is deliberately a DETERMINISTIC
# digit policy (`argmax`), NOT `rank_k`: the trainer's pool has no K-chain
# structure, so it always pins at rank 0 (== argmax). A `rank_k` P* would make
# the S2 "matched" cell train at argmax but eval rank-diverse — i.e. NOT truly
# matched on the digit axis, muddying the very distribution-fidelity claim S2
# tests. Keeping P*'s digit policy deterministic makes train and eval identical
# on that axis by construction. Re-point this to the S1 winner before running
# S2 (prefer a softmax/argmax digit policy for the same reason); everything
# below derives from these two dicts.
# --------------------------------------------------------------------------
P0 = {"cell_policy": "uniform", "digit_policy": "softmax"}
PSTAR = {"cell_policy": "mrv", "digit_policy": "argmax"}

# S2 matched-policy trainer: trains WITH P* so the pool's state distribution
# matches the inference policy. Trained at 1K steps to pair with the base_1k
# mismatch control (where policy effects are largest).
_add_train(
    "train_pstar_1k", "S2", {
        "steps": 1000, "eval_every": 50,
        **PSTAR,
    }, n_seeds=3,
    note=f"S2 matched-training: train WITH P*={PSTAR}. PLACEHOLDER P* — "
         "re-point to the S1 winner before running. Eval both P0 and P* below.",
)


# --------------------------------------------------------------------------
# S1 — decision-policy scan (~10 selected combos, NOT the full cross).
# Eval-only on {baseline (2K), base_1k}. 1 seed (exploratory). The first combo
# (uniform/softmax) is the SANITY GATE: it must reproduce the frozen ckpt's
# no-flag eval. Cell axis: uniform|mrv|min_entropy|max_entropy; digit axis:
# softmax|argmax|rank_k. Selected combos probe each axis + the greedy corner
# and the rank_k diversity fix.
# --------------------------------------------------------------------------
S1_COMBOS: list[tuple[str, str, str]] = [
    # (tag, cell_policy, digit_policy)
    ("baseline_pol", "uniform", "softmax"),   # SANITY GATE (== frozen eval)
    ("mrv_softmax", "mrv", "softmax"),
    ("mrv_argmax", "mrv", "argmax"),
    ("mrv_rankk", "mrv", "rank_k"),
    ("minent_softmax", "min_entropy", "softmax"),
    ("minent_argmax", "min_entropy", "argmax"),   # greedy corner (degenerate-diversity)
    ("minent_rankk", "min_entropy", "rank_k"),    # the diversity fix for greedy
    ("maxent_softmax", "max_entropy", "softmax"),  # control: should be bad
    ("uniform_argmax", "uniform", "argmax"),
    ("uniform_rankk", "uniform", "rank_k"),
]

for base in ("baseline", "base_1k"):
    for tag, cp, dp in S1_COMBOS:
        is_gate = (tag == "baseline_pol")
        _add_eval(
            f"s1_{base}_{tag}", "S1", base,
            {"cell_policy": cp, "digit_policy": dp},
            n_seeds=(3 if is_gate else 1),
            note=("SANITY GATE: uniform/softmax must reproduce the frozen "
                  f"{base} no-flag eval." if is_gate
                  else f"cell={cp} digit={dp} on {base}."),
        )


# --------------------------------------------------------------------------
# S2 — matched vs mismatched 2x2. train-P0 = base_1k (already trained above,
# eval it under P0 and P*); train-P* = train_pstar_1k (eval under P0 and P*).
# P0 and P* are defined once above (train_pstar_1k derives from the same dicts).
# 3 seeds (table rows). The train-P0/eval-P0 and train-P0/eval-P* cells reuse
# the base_1k checkpoint under the two policies; the train-P*/eval-* cells use
# train_pstar_1k.
# --------------------------------------------------------------------------
_add_eval("s2_trainP0_evalP0", "S2", "base_1k", dict(P0), n_seeds=3,
          note="train P0 / eval P0 (the baseline cell of the 2x2).")
_add_eval("s2_trainP0_evalPstar", "S2", "base_1k", dict(PSTAR), n_seeds=3,
          note="train P0 / eval P* (mismatch control — bolt P* onto a P0 model).")
_add_eval("s2_trainPstar_evalP0", "S2", "train_pstar_1k", dict(P0), n_seeds=3,
          note="train P* / eval P0 (mismatch control — other direction).")
_add_eval("s2_trainPstar_evalPstar", "S2", "train_pstar_1k", dict(PSTAR), n_seeds=3,
          note="train P* / eval P* (the MATCHED run).")


# --------------------------------------------------------------------------
# S3 — backtracking policies. Eval-only (phase 1) on {baseline, base_1k}.
# root (sanity gate) | last | geometric(0.5) | uniform_depth | last+negate.
# 3 seeds on the table rows; the S3 table reports unsound_negation_rate.
# --------------------------------------------------------------------------
S3_POLICIES: list[tuple[str, dict]] = [
    ("root", {"backtrack": "root"}),                     # SANITY GATE
    ("last", {"backtrack": "last"}),
    ("geom05", {"backtrack": "geometric", "geometric_p": 0.5}),
    ("unifdepth", {"backtrack": "uniform_depth"}),
    ("lastneg", {"backtrack": "last+negate"}),
]
for base in ("baseline", "base_1k"):
    for tag, flags in S3_POLICIES:
        is_gate = (tag == "root")
        _add_eval(
            f"s3_{base}_{tag}", "S3", base, dict(flags),
            n_seeds=3,
            note=("SANITY GATE: backtrack=root must reproduce the frozen "
                  f"{base} no-flag eval." if is_gate
                  else f"backtrack={flags['backtrack']} on {base}."),
        )


# --------------------------------------------------------------------------
# S4 — policy gain vs model strength. 2-3 best (policy, backtrack) combos +
# baseline policy, evaluated across the training-budget axis {1K, 2K, 4K}.
#   1K = base_1k (E2), 2K = baseline (E1), 4K = base_4k (E2).
# Combos (PLACEHOLDER best set — re-point after S1/S3): P0/root (baseline
# reference), P*/root, P*/last, P0/last. 1 seed each (figure, exploratory).
# --------------------------------------------------------------------------
S4_CKPTS = [("1k", "base_1k"), ("2k", "baseline"), ("4k", "base_4k")]
S4_COMBOS: list[tuple[str, dict]] = [
    ("p0_root", {**P0, "backtrack": "root"}),
    ("pstar_root", {**PSTAR, "backtrack": "root"}),
    ("pstar_last", {**PSTAR, "backtrack": "last"}),
    ("p0_last", {**P0, "backtrack": "last"}),
]
for budget, base in S4_CKPTS:
    for tag, flags in S4_COMBOS:
        _add_eval(
            f"s4_{budget}_{tag}", "S4", base, dict(flags),
            n_seeds=1,
            note=f"S4 budget={budget} combo={tag} on {base}.",
        )


# --------------------------------------------------------------------------
# Ordered sub-study grouping for `list` output.
# --------------------------------------------------------------------------
STUDY_ORDER = ["TRAIN", "S1", "S2", "S3", "S4"]
STUDY_LABEL = {
    "TRAIN": "TRAIN  checkpoints E2 owns (base_1k / base_4k / matched-policy)",
    "S1": "S1  decision-policy scan (eval-only; uniform/softmax is the gate)",
    "S2": "S2  matched-vs-mismatched 2x2 (train P0/P* x eval P0/P*)",
    "S3": "S3  backtracking policies (eval-only; root is the gate)",
    "S4": "S4  policy gain vs model strength (1K/2K/4K x best combos)",
}


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

_flag = _common.flag


def input_checkpoint(config_name: str, seed: int) -> str:
    """Fixed volume path of the checkpoint an eval config consumes."""
    inp = CONFIGS[config_name]["input"]
    subdir = INPUT_SUBDIR[inp]
    return f"/checkpoints/{subdir}/{inp}_seed{seed}.pt"


def output_name(config_name: str, seed: int) -> str:
    """Deterministic ckpt-name for a config's landed eval.json.

    Train configs: `<config>_seed<N>` (checkpoint + its own eval).
    Eval configs:  `<config>__on__<input>_seed<N>` (the __on__ convention).
    """
    cfg = CONFIGS[config_name]
    if cfg["kind"] == "train":
        return f"{config_name}_seed{seed}"
    return f"{config_name}__on__{cfg['input']}_seed{seed}"


def effective_train_flags(config_name: str) -> dict[str, object]:
    """Effective run.py flag dict for a train config (DEFAULTS + overrides)."""
    cfg = CONFIGS[config_name]
    eff = dict(TRAIN_DEFAULTS)
    eff.update(cfg["overrides"])
    return {k: v for k, v in eff.items() if v is not DROP}


def effective_eval_flags(config_name: str) -> dict[str, object]:
    """Effective eval_only.py flag dict (EVAL_DEFAULTS + per-config deviations)."""
    cfg = CONFIGS[config_name]
    eff = dict(EVAL_DEFAULTS)
    eff.update(cfg["eval_flags"])
    return {k: v for k, v in eff.items() if v is not DROP}


def _emit_bool(parts: list[str], name: str, val: bool, default: bool) -> None:
    """Emit a bool flag only when it DIFFERS from the entrypoint default."""
    if bool(val) == bool(default):
        return
    if val:
        parts.append(_flag(name))
    else:
        parts.append(f"--no-{name.replace('_', '-')}")


# Entrypoint bool defaults (for emitting only non-default bool flags).
_EVAL_BOOL_DEFAULT = {"augment": True, "estimate_sequential": False,
                      "learn_negation": False}


def launch_command(config_name: str, seed: int) -> str:
    """Build the `uv run modal run --detach ...` command for (config, seed)."""
    cfg = CONFIGS[config_name]
    if cfg["kind"] == "train":
        return _common.launch_command(
            config_name, seed, effective_train_flags(config_name),
            ckpt_subdir=CKPT_SUBDIR, entrypoint=RUN_ENTRYPOINT,
            flag_order=TRAIN_FLAG_ORDER, skip_if_done=True,
        )
    # kind == "eval": target eval_only.py with a fixed --checkpoint.
    eff = effective_eval_flags(config_name)
    ckpt = input_checkpoint(config_name, seed)
    out = output_name(config_name, seed)
    parts = [f"uv run modal run --detach {EVAL_ENTRYPOINT}"]
    parts.append(f"{_flag('checkpoint')} {ckpt}")
    ordered = list(EVAL_FLAG_ORDER) + [k for k in eff if k not in EVAL_FLAG_ORDER]
    emitted: set[str] = set()
    for name in ordered:
        if name in emitted or name not in eff:
            continue
        emitted.add(name)
        val = eff[name]
        if name in BOOL_FLAGS:
            _emit_bool(parts, name, val, _EVAL_BOOL_DEFAULT.get(name, False))
        else:
            parts.append(f"{_flag(name)} {val}")
    parts.append(f"{_flag('ckpt_subdir')} {CKPT_SUBDIR}")
    parts.append(f"{_flag('ckpt_name')} {out}")
    parts.append("--skip-if-done")
    return " \\\n    ".join(parts)


def iter_runs(prefix: str = ""):
    """Yield (config_name, seed) for every run, optionally filtered by prefix."""
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
# Volume querying (status / remaining). Non-aborting status (matches E1/E3).
# Eval configs land under followups/e2 with __on__ names; train configs land
# under followups/e2 with <config>_seed<N>. Both share the same done-set.
# --------------------------------------------------------------------------

def _try_volume_done_set() -> tuple[set[str] | None, str | None]:
    try:
        return _common.volume_done_set(CKPT_SUBDIR), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _matches(name: str, study: str, prefix: str) -> bool:
    return (not prefix) or (prefix.lower() in name.lower()
                            or prefix.lower() in study.lower())


def _print_list(prefix: str = "") -> None:
    print("# " + "=" * 72, flush=True)
    print("# E2 (search_process) launch commands.", flush=True)
    print("#", flush=True)
    print("# PREREQUISITES:", flush=True)
    print("#   - E1 baseline at /checkpoints/followups/e1/baseline_seed<N>.pt", flush=True)
    print("#   - Train E2's own checkpoints FIRST (TRAIN group: base_1k /", flush=True)
    print("#     base_4k / train_pstar_1k) before the eval-only S1/S2/S3/S4.", flush=True)
    print("#", flush=True)
    print("# 1. SANITY GATES FIRST: s1_*_baseline_pol (uniform/softmax) and", flush=True)
    print("#    s3_*_root (backtrack=root) must reproduce the frozen ckpt's", flush=True)
    print("#    no-flag eval before any policy result is interpreted.", flush=True)
    print("# 2. Launch each command INDIVIDUALLY (one `modal run --detach` per", flush=True)
    print("#    run). Do NOT wrap these in a shell loop.", flush=True)
    print("# 3. Artifacts land under /checkpoints/followups/e2/.", flush=True)
    print("# 4. Every command carries --skip-if-done (idempotent re-launch).", flush=True)
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
            if cfg["kind"] == "eval":
                print(f"# {name}  (eval on {cfg['input']}, {cfg['n_seeds']} "
                      f"seed(s)){note}", flush=True)
            else:
                print(f"# {name}  (train, {cfg['n_seeds']} seed(s)){note}",
                      flush=True)
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
    print(f"# E2 volume status  (/{CKPT_SUBDIR}/, {len(done)} eval.json files)"
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
    print("# E2 REMAINING launch commands (missing eval.json only).", flush=True)
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
                        help="optional prefix/substring filter (e.g. s1, s3, train)")
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
