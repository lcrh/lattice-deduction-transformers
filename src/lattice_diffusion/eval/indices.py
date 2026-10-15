"""Named index lists for evaluation.

Each list is a deterministic function of (split size, seed) and is checked
against a recorded SHA-256, so every run walks exactly the same puzzles in
the same order and silent drift (e.g. a numpy change) fails loudly.

- `sudoku_test_perm`: random permutation of the full Sudoku-Extreme test
  split. Sequential evals walk it prefix by prefix; the full split is this
  list evaluated with `NoStop`.
- `sudoku_val`: 10k puzzles from the train split, held out from every
  training pool (`SudokuExtremeDataset` excludes them) and used for choosing
  defaults, never for reported numbers.
- `maze_test`: the 1,000-puzzle Maze-Hard test split.
- `snowflake_varied_test` / `snowflake_varied_val`: the varied-topology
  Snowflake test and validation splits (the paper's Snowflake dataset);
  `snowflake_test` / `snowflake_val` are the canonical-topology ones.
  Their sizes are only known once the data is generated, so they start
  unpinned: the caller supplies the split size and each run directory
  records the digest it used (runner.py), so a regenerated dataset is
  caught on resume. `experiments/snowflake/gen_data.py` prints the values
  to pin after generating.
  TODO: pin snowflake_varied_{test,val} once the camera-ready data exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndexListSpec:
    task: str
    split: str
    split_size: int | None  # None = unpinned (see module docstring)
    seed: int
    length: int | None      # None = full permutation
    sha256: str | None      # None = unpinned


SPECS: dict[str, IndexListSpec] = {
    "sudoku_test_perm": IndexListSpec(
        task="sudoku", split="test", split_size=422_786, seed=20261015,
        length=None,
        sha256="f6c60663a53881e5627ee960e9b5f95514d84f22f71ac124b2e8a710ca4bdb50",
    ),
    "sudoku_val": IndexListSpec(
        task="sudoku", split="train", split_size=3_831_994, seed=20261016,
        length=10_000,
        sha256="c46dc1bb61b30cfdaf0df93da5ba75b388741f8968647a498bb8ad0f2ede32ad",
    ),
    "maze_test": IndexListSpec(
        task="maze", split="test", split_size=1_000, seed=20261017,
        length=None,
        sha256="31a7a6ad8ad29ea1c818d39ab870a4b562dc0d70ec8f9c9c963383d4272b0250",
    ),
    "snowflake_test": IndexListSpec(
        task="snowflake", split="test", split_size=None, seed=20261018,
        length=None, sha256=None,
    ),
    "snowflake_val": IndexListSpec(
        task="snowflake", split="val", split_size=None, seed=20261019,
        length=None, sha256=None,
    ),
    "snowflake_varied_test": IndexListSpec(
        task="snowflake", split="varied_test", split_size=None, seed=20261020,
        length=None, sha256=None,
    ),
    "snowflake_varied_val": IndexListSpec(
        task="snowflake", split="varied_val", split_size=None, seed=20261021,
        length=None, sha256=None,
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


def _digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype="<i8").tobytes()).hexdigest()


def generate(spec: IndexListSpec, split_size: int | None = None) -> np.ndarray:
    n = spec.split_size if spec.split_size is not None else split_size
    perm = np.random.default_rng(spec.seed).permutation(n).astype(np.int64)
    return perm if spec.length is None else perm[:spec.length]


def is_pinned(name: str) -> bool:
    return SPECS[name].split_size is not None


def load(name: str, split_size: int | None = None) -> IndexList:
    """Build a named list. Pass the actual split size when available so a
    dataset that changed size is caught here rather than mid-run; it is
    required for unpinned lists."""
    spec = SPECS[name]
    if spec.split_size is None:
        if split_size is None:
            raise ValueError(f"{name} is unpinned; pass the dataset's split size")
        return IndexList(name=name, task=spec.task, split=spec.split,
                         indices=generate(spec, split_size), split_size=split_size)
    if split_size is not None and split_size != spec.split_size:
        raise ValueError(f"{name}: dataset split has {split_size} items, "
                         f"index list expects {spec.split_size}")
    indices = generate(spec)
    if _digest(indices) != spec.sha256:
        raise ValueError(f"{name}: generated indices do not match the recorded "
                         f"checksum; refusing to evaluate on a different list")
    return IndexList(name=name, task=spec.task, split=spec.split, indices=indices,
                     split_size=spec.split_size)


def filter_tag(index_filter: dict) -> str:
    """Short stable name for an index filter, e.g. orders=7,8,9,10."""
    return ";".join(f"{k}={','.join(map(str, v)) if isinstance(v, list) else v}"
                    for k, v in sorted(index_filter.items()))


def explicit(task: str, split: str, indices, name: str = "explicit") -> IndexList:
    """An ad-hoc list (e.g. a portfolio failure set), evaluated in the given order."""
    arr = np.asarray(indices, dtype=np.int64)
    return IndexList(name=f"{name}-{_digest(arr)[:12]}", task=task, split=split, indices=arr)


def digest(index_list: IndexList) -> str:
    return _digest(index_list.indices)
