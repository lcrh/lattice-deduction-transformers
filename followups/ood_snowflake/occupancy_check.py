"""E4 occupancy check — quantify the Snowflake positional confound.

Snowflake placement is deterministic and hub-centered: every hex cell maps to a
FIXED covering-grid slot via `cell_to_grid_idx`, regardless of puzzle order.
Snowflakes grow outward with order, so training on small orders leaves the outer
covering-grid slots seen only as "absent" — their learned positional embeddings
would be effectively untrained when a larger order first activates them. A
transfer failure would then measure that under-training, not constraint
generalization.

This script measures the SIZE of that confound before any training. For each
order it computes which of the 150 covering-grid slots are ever active
(in_puzzle_mask True) at that order, then reports:
  - a per-order occupancy table (how many slots active at each order), and
  - the confound counts: how many slots become active at orders 7-8 but are
    NEVER active at orders <=6 (and similarly <=5).

Standalone: reads a parquet (or JSON) puzzle file directly; no Modal, no GPU.

Usage:
    python followups/ood_snowflake/occupancy_check.py --data data/snowflake_train.parquet
    python followups/ood_snowflake/occupancy_check.py --data /path/to/snowflake_test.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import the covering-grid embedding + loader from the experiment package so we
# measure occupancy through EXACTLY the same code path training uses.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.snowflake.data import (  # noqa: E402
    SEQ_LEN,
    _load_puzzles,
    puzzle_to_state,
)


def compute_occupancy(puzzles: list[dict]) -> dict[int, np.ndarray]:
    """Return {order n -> bool array [SEQ_LEN]} of slots ever active at that order."""
    occ: dict[int, np.ndarray] = {}
    for rec in puzzles:
        n = int(rec["n"])
        _x, _y, mask = puzzle_to_state(rec)
        if n not in occ:
            occ[n] = np.zeros((SEQ_LEN,), dtype=bool)
        occ[n] |= mask
    return occ


def _union(occ: dict[int, np.ndarray], orders) -> np.ndarray:
    acc = np.zeros((SEQ_LEN,), dtype=bool)
    for n in orders:
        if n in occ:
            acc |= occ[n]
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", required=True,
        help="Path to a snowflake parquet file/dir (or JSON) with per-row `n`.",
    )
    args = ap.parse_args()

    p = Path(args.data)
    if not p.exists():
        print(
            f"ERROR: data path {args.data!r} not found.\n"
            "Point --data at a generated snowflake parquet, e.g. a file pulled "
            "from the `lattice-diffusion-data` Modal volume "
            "(snowflake_train.parquet / snowflake_test.parquet).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        puzzles = _load_puzzles(str(p))
    except Exception as e:  # pragma: no cover - depends on optional pyarrow
        print(
            f"ERROR: failed to load {args.data!r}: {e}\n"
            "If this is a parquet file, ensure pyarrow is installed "
            "(`uv pip install pyarrow`).",
            file=sys.stderr,
        )
        sys.exit(1)

    occ = compute_occupancy(puzzles)
    orders = sorted(occ.keys())
    n_per_order: dict[int, int] = {}
    for rec in puzzles:
        n_per_order[int(rec["n"])] = n_per_order.get(int(rec["n"]), 0) + 1

    print("=" * 60)
    print(f"Snowflake occupancy over {SEQ_LEN} covering-grid slots")
    print(f"  data: {args.data}")
    print(f"  puzzles: {len(puzzles)}   orders present: {orders}")
    print("=" * 60)
    print(f"  {'order':>5}  {'#puzzles':>9}  {'#active_slots':>13}  "
          f"{'#new_vs_smaller':>15}")
    seen_smaller = np.zeros((SEQ_LEN,), dtype=bool)
    for n in orders:
        active = occ[n]
        n_active = int(active.sum())
        # Slots active at this order but at NO strictly-smaller order.
        new_here = int((active & ~seen_smaller).sum())
        print(f"  {n:>5}  {n_per_order[n]:>9}  {n_active:>13}  {new_here:>15}")
        seen_smaller |= active
    print("-" * 60)

    # Confound: slots active at orders 7-8 but NEVER active at <=k.
    for k in (6, 5):
        small = _union(occ, [n for n in orders if n <= k])
        big = _union(occ, [n for n in orders if n in (7, 8)])
        confound = int((big & ~small).sum())
        print(
            f"  slots active at orders 7-8 but NEVER at <= {k}: "
            f"{confound} / {SEQ_LEN}"
        )
    # Also the natural transfer boundaries used by the E4 configs.
    for train_max, test_orders in ((5, (6, 7, 8)), (6, (7, 8))):
        small = _union(occ, [n for n in orders if n <= train_max])
        big = _union(occ, [n for n in orders if n in test_orders])
        confound = int((big & ~small).sum())
        print(
            f"  [e4_leq{train_max}] slots active in test orders {test_orders} "
            f"but NEVER in train orders <= {train_max}: {confound} / {SEQ_LEN}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
