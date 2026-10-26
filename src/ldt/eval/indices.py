"""Named index lists for evaluation.

Each list is a seeded permutation of a dataset split, checked against a
recorded SHA-256, so every run walks the same puzzles in the same order.

Snowflake splits are only sized once the data is generated, so their lists
start unpinned: the caller supplies the split size, and each run directory
records the digest it used, so a regenerated dataset is caught on resume.
`experiments/snowflake/gen_data.py` prints the values to pin.

A list can exclude indices before permuting (`exclude`): `sudoku_holdout`
draws from train-split puzzles that no portfolio model trained on (the
1,000-puzzle pools for data seeds 42-52) and that are not in `sudoku_val`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndexListSpec:
    task: str
    split: str
    split_size: int | None  # None = unpinned
    seed: int
    length: int | None      # None = full permutation
    sha256: str | None      # None = unpinned
    exclude: Callable[[int], np.ndarray] | None = None   # split size -> indices to leave out

    @property
    def pinned(self) -> bool:
        return self.split_size is not None


PORTFOLIO_DATA_SEEDS = tuple(range(42, 53))


def sudoku_training_puzzles(n_total: int) -> np.ndarray:
    """sudoku_val plus the 1,000-puzzle training pools for the portfolio's data seeds."""
    from ldt.data.sudoku_extreme import SudokuExtremeConfig, select_pool

    val = load("sudoku_val", split_size=n_total).indices
    pools = [select_pool(n_total, SudokuExtremeConfig(n_puzzles=1000, seed=seed))[0]
             for seed in PORTFOLIO_DATA_SEEDS]
    return np.unique(np.concatenate([val, *pools]))


SPECS: dict[str, IndexListSpec] = {
    # Full Sudoku-Extreme test split; sequential evals walk it prefix by prefix.
    "sudoku_test_perm": IndexListSpec(
        task="sudoku", split="test", split_size=422_786, seed=20261015,
        length=None,
        sha256="f6c60663a53881e5627ee960e9b5f95514d84f22f71ac124b2e8a710ca4bdb50",
    ),
    # Held out from every training pool; for choosing defaults, never for reported numbers.
    "sudoku_val": IndexListSpec(
        task="sudoku", split="train", split_size=3_831_994, seed=20261016,
        length=10_000,
        sha256="c46dc1bb61b30cfdaf0df93da5ba75b388741f8968647a498bb8ad0f2ede32ad",
    ),
    # Train-split puzzles no portfolio model saw; for choosing the portfolio recipe.
    "sudoku_holdout": IndexListSpec(
        task="sudoku", split="train", split_size=3_831_994, seed=20261022,
        length=100_000, exclude=sudoku_training_puzzles,
        sha256="4b557d8529f206d9d2cd731e3a8424ffdbba79f6f5ff15eadd357b3278fdd3e9",
    ),
    "maze_test": IndexListSpec(
        task="maze", split="test", split_size=1_000, seed=20261017,
        length=None,
        sha256="31a7a6ad8ad29ea1c818d39ab870a4b562dc0d70ec8f9c9c963383d4272b0250",
    ),
    # Snowflake splits that also keep puzzles with givens added back (the base config's list).
    "snowflake_test": IndexListSpec(
        task="snowflake", split="test", split_size=4292, seed=20261020,
        length=None,
        sha256="0fb2e04aa36086d9735d6a2e70dfebab1163fb05ca715cb8e33623ee736bc0cd",
    ),
    "snowflake_val": IndexListSpec(
        task="snowflake", split="val", split_size=1761, seed=20261021,
        length=None,
        sha256="e4547df0f6af336ee555cc061019d386c16687b9c1eee79c1f47fc9e2ab2b591",
    ),
    # Snowflake-Extreme splits (gen_data.py --extreme).
    "snowflake_extreme_test": IndexListSpec(
        task="snowflake", split="extreme_test", split_size=951, seed=20261025,
        length=None,
        sha256="980ea6b00e7cd82a611c525898c1d8f29974f272bc25d3723ed334728681ec82",
    ),
    "snowflake_extreme_val": IndexListSpec(
        task="snowflake", split="extreme_val", split_size=476, seed=20261026,
        length=None,
        sha256="fd31d0d384eef31ac621eb095384f17804c8b395157083ebf87858fb1e4ca7ed",
    ),
}


@dataclass(frozen=True)
class IndexList:
    name: str
    task: str
    split: str
    indices: np.ndarray     # absolute dataset indices, int64, in evaluation order
    split_size: int | None = None   # size of the split the indices point into

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def digest(self) -> str:
        return digest(self.indices)


def digest(indices: np.ndarray) -> str:
    """SHA-256 of the indices as little-endian int64."""
    return hashlib.sha256(np.ascontiguousarray(indices, dtype="<i8").tobytes()).hexdigest()


def generate(spec: IndexListSpec, split_size: int | None = None) -> np.ndarray:
    """Indices for `spec`; `split_size` is used only when the spec is unpinned."""
    n = spec.split_size if spec.pinned else split_size
    pool = np.arange(n, dtype=np.int64)
    if spec.exclude is not None:
        pool = np.setdiff1d(pool, spec.exclude(n), assume_unique=True)
    perm = pool[np.random.default_rng(spec.seed).permutation(len(pool))]
    return perm if spec.length is None else perm[:spec.length]


def load(name: str, split_size: int | None = None) -> IndexList:
    """Build named list `name`, checked against its recorded SHA-256.

    `split_size` is required for unpinned lists; pass it otherwise to catch a
    resized dataset early.
    """
    spec = SPECS[name]
    if not spec.pinned:
        if split_size is None:
            raise ValueError(f"{name} is unpinned; pass the dataset's split size")
        return IndexList(name=name, task=spec.task, split=spec.split,
                         indices=generate(spec, split_size), split_size=split_size)
    if split_size is not None and split_size != spec.split_size:
        raise ValueError(f"{name}: dataset split has {split_size} items, "
                         f"index list expects {spec.split_size}")
    indices = generate(spec)
    if digest(indices) != spec.sha256:
        raise ValueError(f"{name}: generated indices do not match the recorded "
                         f"checksum; refusing to evaluate on a different list")
    return IndexList(name=name, task=spec.task, split=spec.split, indices=indices,
                     split_size=spec.split_size)


def filter_tag(index_filter: dict) -> str:
    """Short stable name for an index filter, e.g. orders=7,8,9,10."""
    return ";".join(f"{k}={','.join(map(str, v)) if isinstance(v, list) else v}"
                    for k, v in sorted(index_filter.items()))


def explicit(task: str, split: str, indices) -> IndexList:
    """An ad-hoc list (e.g. a portfolio failure set), evaluated in the given order."""
    arr = np.asarray(indices, dtype=np.int64)
    return IndexList(name=f"explicit-{digest(arr)[:12]}", task=task, split=split, indices=arr)
