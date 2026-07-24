"""E6 latent-carry run matrix and idempotent Modal launch commands.

Usage:
    uv run python followups/latent_carry/configs.py list
    uv run python followups/latent_carry/configs.py status
    uv run python followups/latent_carry/configs.py remaining
"""

from __future__ import annotations

import argparse
import sys

from followups import _common


CKPT_SUBDIR = "followups/e6"
VOLUME_NAME = _common.VOLUME_NAME

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
    # Fixed-frame regime: each dataset sample is augmented once when it
    # enters the pool; dpll_step never resamples its frame.
    "augment": False,
    "data_augment_digit_perm": True,
    "data_augment_dihedral": True,
}

CONFIGS: dict[str, dict] = {
    "baseline": {
        "n_seeds": 3,
        "overrides": {"carry_latent": "off"},
        "note": "Lattice-only matched fixed-frame baseline.",
    },
    "carry_h": {
        "n_seeds": 3,
        "overrides": {"carry_latent": "h"},
        "note": "Carry the ordinary final head-feeding hidden.",
    },
    "carry_z": {
        "n_seeds": 3,
        "overrides": {"carry_latent": "z"},
        "note": "Carry a separate TRM-style scratchpad.",
    },
    "zero_carry_z": {
        "n_seeds": 3,
        "overrides": {"carry_latent": "zero_z"},
        "note": "Scratchpad architecture with carry zeroed at every boundary.",
    },
}

FLAG_ORDER = (
    "steps", "n_train_puzzles", "n_eval_puzzles",
    "num_layers", "dim", "n_loops", "supervise",
    "carry_latent", "augment",
    "data_augment_digit_perm", "data_augment_dihedral",
    "softmax_loss_weight", "conflict_loss_weight",
    "bce_pos_mult", "bce_neg_mult", "eval_every",
)


def effective_flags(config_name: str) -> dict[str, object]:
    flags = dict(BASE_DEFAULTS)
    flags.update(CONFIGS[config_name]["overrides"])
    return flags


def launch_command(config_name: str, seed: int) -> str:
    return _common.launch_command(
        config_name,
        seed,
        effective_flags(config_name),
        ckpt_subdir=CKPT_SUBDIR,
        flag_order=FLAG_ORDER,
        skip_if_done=True,
    )


def iter_runs():
    for name, cfg in CONFIGS.items():
        for seed in range(cfg["n_seeds"]):
            yield name, seed


def _done_set() -> set[str]:
    return _common.volume_done_set(CKPT_SUBDIR)


def _print_list() -> None:
    print("# E6 latent-carry launch commands (launch individually; no shell loop).")
    print("# Run baseline_seed0 as the fixed-frame sanity gate before the rest.\n")
    for name, cfg in CONFIGS.items():
        print(f"# {name}: {cfg['note']}")
        for seed in range(cfg["n_seeds"]):
            print(launch_command(name, seed))
            print()


def _print_status(*, remaining: bool = False) -> None:
    try:
        done = _done_set()
    except Exception as exc:  # noqa: BLE001
        if remaining:
            print(f"[remaining] volume query failed: {type(exc).__name__}: {exc}")
            sys.exit(1)
        done = set()
        print(f"# [status] volume unavailable: {type(exc).__name__}: {exc}")

    total = landed = 0
    for name, cfg in CONFIGS.items():
        missing = [
            seed for seed in range(cfg["n_seeds"])
            if f"{name}_seed{seed}" not in done
        ]
        total += cfg["n_seeds"]
        landed += cfg["n_seeds"] - len(missing)
        if remaining:
            if missing:
                print(f"# {name}: missing {missing}")
                for seed in missing:
                    print(launch_command(name, seed))
                    print()
        else:
            print(f"{name}: {cfg['n_seeds'] - len(missing)}/{cfg['n_seeds']} done")
    if not remaining:
        print(f"overall: {landed}/{total} done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "status", "remaining"))
    args = parser.parse_args()
    if args.command == "list":
        _print_list()
    elif args.command == "status":
        _print_status()
    else:
        _print_status(remaining=True)


if __name__ == "__main__":
    main()
