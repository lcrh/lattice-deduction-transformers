"""E4 (ood_snowflake) run matrix — the single source of truth.

Order-transfer OOD generalization on Snowflake Sudoku: train on small orders,
evaluate on strictly larger, never-seen orders. A soft distribution-shift
condition retains sparse training support on every order. The README's design
table is encoded here as data: config-name -> {flag overrides} + n_seeds.

Every config matches the standard Snowflake config
(`experiments/snowflake/run.py` defaults) EXCEPT the order filter, the reduced
train-set size (`--n-train-puzzles 500`), and translation augmentation
(`--translate-aug`) — which is ON for ALL configs, including the in-distribution
control e4_all. `e4_shift95` additionally fixes the per-order composition of
that 500-puzzle subset. `--steps` is left at the run.py default (4000).

Each run is a Modal entrypoint launched with:

    uv run modal run --detach experiments/snowflake/run.py -- <flags>

Checkpoints land at the deterministic exchange path
`/checkpoints/followups/e4/<config>_seed<N>.pt` (+ .eval.json / .eval.jsonl /
.train_curve.jsonl) via `--ckpt-subdir followups/e4 --ckpt-name <config>_seed<N>`.

CLI:
    uv run python followups/ood_snowflake/configs.py list          # all configs
    uv run python followups/ood_snowflake/configs.py list e4_leq5  # filter substring
    uv run python followups/ood_snowflake/configs.py status        # done N/M per config
    uv run python followups/ood_snowflake/configs.py remaining     # cmds for missing only

Runs are STATEFUL / idempotent: every emitted launch command carries
`--skip-if-done`, so a whole-sweep re-launch executes only the (config, seed)
pairs whose `.eval.json` has not landed yet. `status` / `remaining` query the
Modal volume to report / target the missing pairs.

This module is importable: `collect.py` and `plot_order_transfer.py` read
`CONFIGS` for the expected per-config order filters (train_orders / eval_orders)
and the OOD boundary (max train order).
"""

from __future__ import annotations

import argparse
import sys

# Absolute package imports rooted at the repo root. Scripts are run from the
# repo root (`uv run python followups/ood_snowflake/configs.py`), so cwd is on
# sys.path and `followups` resolves as a package (every dir has __init__.py).
from followups import _common  # shared volume / flag helpers

# --------------------------------------------------------------------------
# Defaults applied to every run unless a config overrides them.
#   - steps: NOT emitted (left at the run.py default 4000) — README pins the
#     standard Snowflake hyperparameters; only the order filter varies.
#   - n_train_puzzles 500 (README).
#   - translate_aug ON for ALL configs (transfer AND control) so the
#     positional confound is mitigated uniformly.
# --------------------------------------------------------------------------

BASE_DEFAULTS: dict[str, object] = {
    "n_train_puzzles": 500,
    "translate_aug": True,   # emitted as the boolean flag --translate-aug
}

# run.py default for steps (documented here for collect.py; not emitted as a
# flag so the standard default is used verbatim).
DEFAULT_STEPS = 4000

CKPT_SUBDIR = "followups/e4"
VOLUME_NAME = _common.VOLUME_NAME
RUN_ENTRYPOINT = "experiments/snowflake/run.py"

# Boolean flags: emitted as a bare `--flag` when truthy, omitted otherwise.
BOOL_FLAGS = {"translate_aug", "use_rope"}


# --------------------------------------------------------------------------
# The run matrix. Each entry:
#   name -> {"train_orders": str, "eval_orders": str, "n_seeds": int,
#            "overrides": {...extra flag overrides...}, "optional": bool,
#            "note": str}
# `train_orders` / `eval_orders` are comma-separated strings passed verbatim
# to --train-orders / --eval-orders. `overrides` carries any deviation beyond
# BASE_DEFAULTS + the order filters (e.g. use_rope for the RoPE variant).
# --------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {}


def _add(name, train_orders, eval_orders, overrides=None, n_seeds=3,
         optional=False, note=""):
    CONFIGS[name] = {
        "train_orders": train_orders,
        "eval_orders": eval_orders,
        "overrides": dict(overrides or {}),
        "n_seeds": n_seeds,
        "optional": optional,
        "note": note,
    }


# --- e4_all: in-distribution control (sanity gate) -------------------------
# Standard held-out-split setting, re-run under the same aug regime. Must
# reproduce the known ~100/100 result before any transfer run is interpreted.
_add(
    "e4_all", "4,5,6,7,8", "4,5,6,7,8",
    n_seeds=3,
    note="CONTROL / SANITY GATE. Standard held-out split, re-run under the "
         "transfer aug regime. Must reproduce ~100/100 before interpreting "
         "transfer runs.",
)

# --- e4_leq5: train {4,5}, test {6,7,8} (+ 9,10 stretch) -------------------
_add(
    "e4_leq5", "4,5", "6,7,8,9,10",
    n_seeds=3,
    note="Transfer from orders <=5. Test 6,7,8 + stretch 9,10 (9-10 require "
         "gen_data.py --n-max 10; absent orders simply yield 0 eval puzzles).",
)

# --- e4_leq6: train {4,5,6}, test {7,8} (+ 9,10) ---------------------------
_add(
    "e4_leq6", "4,5,6", "7,8,9,10",
    n_seeds=3,
    note="Transfer from orders <=6. Test 7,8 + stretch 9,10.",
)

# --- e4_shift95: all orders, but 95% of train examples at orders 4-5 -------
# Exact 500-example mixture: 475 lower-order and 25 higher-order puzzles.
# This distinguishes a severe support-preserving order-distribution shift from
# strict unseen-order extrapolation.
_add(
    "e4_shift95", "4,5,6,7,8", "4,5,6,7,8",
    overrides={"train_order_counts": "4:238,5:237,6:9,7:8,8:8"},
    n_seeds=3,
    note="Soft order-distribution shift: all orders remain in support, but "
         "95% (475/500) of training puzzles are orders 4-5 and only 5% "
         "(25/500) are orders 6-8. Evaluate on the balanced held-out split.",
)

# --- e4_leq5_rope: relative-position control (OPTIONAL) --------------------
# e4_leq5 + RoPE, to compare learned-absolute + translation-aug vs. relative
# encodings on the same transfer split.
_add(
    "e4_leq5_rope", "4,5", "6,7,8,9,10",
    overrides={"use_rope": True},
    n_seeds=3,
    optional=True,
    note="OPTIONAL relative-position control: e4_leq5 + --use-rope. Compare "
         "vs e4_leq5 (learned-absolute) on the same split.",
)


# Ordered grouping for `list` output (comment headers).
CONFIG_ORDER = [
    "e4_all", "e4_leq5", "e4_leq6", "e4_shift95", "e4_leq5_rope",
]


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

_flag = _common.flag  # `some_name` -> `--some-name` (shared helper)


def max_train_order(config_name: str) -> int:
    """Largest order in the config's train filter (the OOD boundary)."""
    toks = [int(t) for t in CONFIGS[config_name]["train_orders"].split(",")
            if t.strip() != ""]
    return max(toks)


def effective_flags(config_name: str) -> dict[str, object]:
    """Return the effective non-order flag dict (BASE_DEFAULTS + overrides).

    Order filters are handled separately by `launch_command` (they always get
    emitted). Used by collect.py to know the expected per-config params.
    """
    cfg = CONFIGS[config_name]
    eff = dict(BASE_DEFAULTS)
    eff.update(cfg["overrides"])
    return eff


def launch_command(config_name: str, seed: int) -> str:
    """Build the `uv run modal run --detach ...` command for (config, seed)."""
    cfg = CONFIGS[config_name]
    eff = effective_flags(config_name)
    parts = [f"uv run modal run --detach {RUN_ENTRYPOINT}"]
    # Deterministic exchange path.
    parts.append(f"{_flag('ckpt_subdir')} {CKPT_SUBDIR}")
    parts.append(f"{_flag('ckpt_name')} {config_name}_seed{seed}")
    parts.append(f"{_flag('seed')} {seed}")
    # Order filters (always emitted).
    parts.append(f"{_flag('train_orders')} {cfg['train_orders']}")
    parts.append(f"{_flag('eval_orders')} {cfg['eval_orders']}")
    # Remaining effective flags in a stable, readable order.
    for name in (
        "n_train_puzzles", "train_order_counts", "translate_aug", "use_rope",
    ):
        if name not in eff:
            continue
        val = eff[name]
        if name in BOOL_FLAGS:
            if val:
                parts.append(_flag(name))     # bare boolean flag
        else:
            parts.append(f"{_flag(name)} {val}")
    # Idempotency: a landed eval.json makes this run a graceful no-op.
    parts.append("--skip-if-done")
    return " \\\n    ".join(parts)


def iter_runs(prefix: str = "", include_optional: bool = True):
    """Yield (config_name, seed) for every run, optionally filtered by prefix."""
    for name in CONFIG_ORDER:
        if name not in CONFIGS:
            continue
        cfg = CONFIGS[name]
        if cfg["optional"] and not include_optional:
            continue
        if prefix and prefix.lower() not in name.lower():
            continue
        for seed in range(cfg["n_seeds"]):
            yield name, seed


# --------------------------------------------------------------------------
# Volume querying (status / remaining).
# --------------------------------------------------------------------------

def _try_volume_done_set() -> tuple[set[str] | None, str | None]:
    """(done-set, None) on success; (None, error-message) on failure.

    Delegates the volume query (and `vol.reload()`) to the shared
    `followups._common.volume_done_set`; degrades gracefully for status /
    remaining when modal is unavailable / unauthenticated / the dir is missing.
    """
    try:
        return _common.volume_done_set(CKPT_SUBDIR), None
    except Exception as e:  # modal not installed / not authed / dir missing
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Subcommands.
# --------------------------------------------------------------------------

def _print_list(prefix: str = "") -> None:
    print("# " + "=" * 72, flush=True)
    print("# E4 (ood_snowflake) launch commands.", flush=True)
    print("#", flush=True)
    print("# PREREQUISITE (orders 9-10 stretch): generate the larger orders", flush=True)
    print("#   first, else those eval orders simply contribute 0 puzzles:", flush=True)
    print("#   uv run modal run --detach experiments/snowflake/gen_data.py --n-max 10", flush=True)
    print("#", flush=True)
    print("# 1. Run the CONTROL / SANITY GATE e4_all FIRST — it must reproduce", flush=True)
    print("#    ~100/100 before any transfer run is interpreted (e4_all_seed0).", flush=True)
    print("# 2. Launch each command INDIVIDUALLY (one `modal run --detach` per", flush=True)
    print("#    run). Do NOT wrap these in a shell loop.", flush=True)
    print("# 3. Checkpoints land at /checkpoints/followups/e4/<config>_seed<N>.pt", flush=True)
    print("# 4. Every command carries --skip-if-done: re-running the sweep only", flush=True)
    print("#    executes (config, seed) pairs whose eval.json has not landed.", flush=True)
    if prefix:
        print(f"#\n# FILTER: only configs matching {prefix!r}.", flush=True)
    print("# " + "=" * 72, flush=True)

    n_cmds = 0
    for name in CONFIG_ORDER:
        if name not in CONFIGS:
            continue
        if prefix and prefix.lower() not in name.lower():
            continue
        cfg = CONFIGS[name]
        opt = "  [OPTIONAL]" if cfg["optional"] else ""
        note = f"  # {cfg['note']}" if cfg["note"] else ""
        print(f"\n# --- {name}{opt}  (train {{{cfg['train_orders']}}} -> "
              f"eval {{{cfg['eval_orders']}}}, {cfg['n_seeds']} seeds) ---",
              flush=True)
        if note:
            print(f"#{note}", flush=True)
        for seed in range(cfg["n_seeds"]):
            print(launch_command(name, seed), flush=True)
            print("", flush=True)
            n_cmds += 1

    print(f"# Total: {n_cmds} launch commands"
          + (f" (filter {prefix!r})" if prefix else "") + ".", flush=True)


def _print_status(prefix: str = "") -> None:
    done, err = _try_volume_done_set()
    print("# E4 (ood_snowflake) status: landed eval.json per config", flush=True)
    if err is not None:
        print(f"# WARNING: could not query the Modal volume ({err}).", flush=True)
        print("#   Reporting the EXPECTED run counts only (0 done assumed).", flush=True)
        done = set()
    print("# " + "-" * 60, flush=True)
    total_done = total_expected = 0
    for name in CONFIG_ORDER:
        if name not in CONFIGS:
            continue
        if prefix and prefix.lower() not in name.lower():
            continue
        cfg = CONFIGS[name]
        n_seeds = cfg["n_seeds"]
        got = sum(1 for s in range(n_seeds) if f"{name}_seed{s}" in done)
        total_done += got
        total_expected += n_seeds
        missing = [s for s in range(n_seeds) if f"{name}_seed{s}" not in done]
        opt = " [OPTIONAL]" if cfg["optional"] else ""
        miss_str = "" if not missing else f"   missing seeds: {missing}"
        print(f"  {name:<16} {got}/{n_seeds} done{opt}{miss_str}", flush=True)
    print("# " + "-" * 60, flush=True)
    tag = " (volume unavailable — counts are expected totals)" if err else ""
    print(f"# Total: {total_done}/{total_expected} runs done{tag}.", flush=True)


def _print_remaining(prefix: str = "") -> None:
    done, err = _try_volume_done_set()
    if err is not None:
        print(f"# ERROR: could not query the Modal volume ({err}).", flush=True)
        print("#   Cannot determine which runs are missing without volume "
              "access.", flush=True)
        print("#   Fix auth / `modal` install, or use `list` to emit ALL "
              "commands", flush=True)
        print("#   (each carries --skip-if-done, so already-done runs no-op).",
              flush=True)
        sys.exit(1)
    print("# E4 (ood_snowflake) REMAINING launch commands (missing eval.json "
          "only).", flush=True)
    print("# Launch each INDIVIDUALLY; each still carries --skip-if-done.",
          flush=True)
    print("# " + "=" * 60, flush=True)
    n_cmds = 0
    for name in CONFIG_ORDER:
        if name not in CONFIGS:
            continue
        if prefix and prefix.lower() not in name.lower():
            continue
        cfg = CONFIGS[name]
        missing = [s for s in range(cfg["n_seeds"])
                   if f"{name}_seed{s}" not in done]
        if not missing:
            continue
        print(f"\n# --- {name}  (missing seeds {missing}) ---", flush=True)
        for seed in missing:
            print(launch_command(name, seed), flush=True)
            print("", flush=True)
            n_cmds += 1
    if n_cmds == 0:
        print("\n# Nothing remaining — all runs have a landed eval.json.",
              flush=True)
    else:
        print(f"# Total remaining: {n_cmds} launch commands.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    p_list = sub.add_parser("list", help="print launch commands (one per run)")
    p_list.add_argument("filter", nargs="?", default="",
                        help="optional substring filter (e.g. e4_leq5)")
    p_status = sub.add_parser("status", help="per-config done N/M (queries volume)")
    p_status.add_argument("filter", nargs="?", default="")
    p_remaining = sub.add_parser(
        "remaining", help="launch commands for missing (config, seed) only")
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
