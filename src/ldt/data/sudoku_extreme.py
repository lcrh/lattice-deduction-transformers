"""On-demand streaming dataset for sudoku powerset training.

Downloads sapientinc/sudoku-extreme from HuggingFace, caches as a .pt file,
and generates powerset samples on the fly with configurable augmentations and
hint types. Fully deterministic given a seed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np
import torch

from ldt.eval import indices as index_lists


@dataclass
class SudokuExtremeConfig:
    cache_dir: str = "data"  # where to cache the HF download (.pt file)
    split: str = "train"  # "train" or "test"
    n_puzzles: int | None = None  # subset size (None = all puzzles in split)
    # Drop the held-out `sudoku_val` puzzles (ldt.eval.indices)
    # from the train split before subset selection, so no training pool ever
    # contains a validation puzzle. Ignored for other splits.
    exclude_val: bool = True
    seed: int = 42

    # Sample type weights (normalized internally)
    zero_hint_weight: float = 0.20
    correct_hint_weight: float = 0.55
    error_hint_weight: float = 0.25

    # Error hint parameters
    error_fill_range: tuple[float, float] = (0.1, 1.0)
    error_rate_range: tuple[float, float] = (0.01, 0.30)

    # Correct hint parameters
    correct_fill_range: tuple[float, float] = (0.0, 1.0)

    # Domain-agnostic augmentations (would work for ARC-AGI too)
    augment_digit_perm: bool = True   # permute digit labels (9!)
    augment_dihedral: bool = True     # full D4: 4 rotations × optional reflection (8 elements)

    # Prefetch
    prefetch_batches: int = 2
    batch_size: int = 2048


def _download_and_cache(cache_dir: str, split: str = "train") -> Path:
    """Download a sudoku-extreme split and cache as .pt file."""
    cache_path = Path(cache_dir) / f"sudoku_extreme_{split}.pt"
    if cache_path.exists():
        return cache_path

    from datasets import load_dataset

    print(f"Downloading sapientinc/sudoku-extreme {split} split...", flush=True)
    ds = load_dataset("sapientinc/sudoku-extreme", split=split)
    n = len(ds)
    print(f"  {n} puzzles. Converting to tensors...", flush=True)

    # Vectorized conversion: string chars -> digit arrays -> one-hot
    char_to_digit = np.zeros(256, dtype=np.uint8)
    for i in range(1, 10):
        char_to_digit[ord(str(i))] = i
    one_hot = np.eye(10, dtype=np.float32)

    q_buf = "".join(ds["question"]).encode("ascii")
    q_arr = np.frombuffer(q_buf, dtype=np.uint8).reshape(n, 81)
    q_digits = char_to_digit[q_arr]
    q_onehot = one_hot[q_digits][:, :, 1:]  # [n, 81, 9]
    x = np.where(q_digits[:, :, None] == 0, np.ones((1, 1, 9), dtype=np.float32), q_onehot)

    a_buf = "".join(ds["answer"]).encode("ascii")
    a_arr = np.frombuffer(a_buf, dtype=np.uint8).reshape(n, 81)
    a_digits = char_to_digit[a_arr]
    y = one_hot[a_digits][:, :, 1:]  # [n, 81, 9]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"x": torch.from_numpy(x).to(torch.uint8), "y": torch.from_numpy(y).to(torch.uint8), "n": n},
        cache_path,
    )
    print(f"  Cached {n} puzzles at {cache_path}", flush=True)
    return cache_path


def load_puzzles(cache_dir: str, split: str,
                 indices: np.ndarray | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) uint8 [len(indices), 81, 9] for absolute indices into a split.

    Memory-maps the cache so pulling a few puzzles out of the 3.8M-puzzle
    train split does not load the whole file.
    """
    data = torch.load(_download_and_cache(cache_dir, split), map_location="cpu",
                      weights_only=True, mmap=True)
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    return data["x"].index_select(0, idx).clone(), data["y"].index_select(0, idx).clone()


def _make_sample(
    question: np.ndarray,
    answer: np.ndarray,
    rng: np.random.Generator,
    sample_type: str,
    cfg: SudokuExtremeConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Generate one powerset sample.

    Args:
        question: [81, 9] uint8 powerset input (givens are one-hot, blanks are all-ones)
        answer: [81, 9] uint8 one-hot solution
        rng: numpy random generator
        sample_type: "zero_hints", "correct_hints", or "error_hints"
        cfg: dataset config

    Returns:
        (x, y, is_sat) where x, y are [81, 9] float32 arrays
    """
    x = question.astype(np.float32)
    y_sol = answer.astype(np.float32)

    if sample_type == "zero_hints":
        return x, y_sol, True

    # Find blank cells (those with all-ones, i.e. sum == 9)
    blank_mask = x.sum(axis=1) > 1.5  # [81], True for blank cells
    blank_indices = np.where(blank_mask)[0]
    n_blanks = len(blank_indices)

    if n_blanks == 0:
        return x, y_sol, True

    if sample_type == "correct_hints":
        lo, hi = cfg.correct_fill_range
        fill_frac = rng.uniform(lo, hi)
        n_fill = int(fill_frac * n_blanks)
        if n_fill > 0:
            to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
            x[to_fill] = y_sol[to_fill]
        return x, y_sol, True

    if sample_type == "error_hints":
        lo, hi = cfg.error_fill_range
        fill_frac = rng.uniform(lo, hi)
        n_fill = max(1, int(fill_frac * n_blanks))
        to_fill = rng.choice(blank_indices, size=n_fill, replace=False)

        # First fill with correct answers
        x[to_fill] = y_sol[to_fill]

        # Then corrupt some
        elo, ehi = cfg.error_rate_range
        n_corrupt = max(1, int(rng.uniform(elo, ehi) * n_fill))
        to_corrupt = rng.choice(to_fill, size=n_corrupt, replace=False)

        for idx in to_corrupt:
            correct_digit = int(y_sol[idx].argmax())
            wrong_digit = rng.choice([d for d in range(9) if d != correct_digit])
            x[idx] = 0.0
            x[idx, wrong_digit] = 1.0

        # UNSAT: target is all-zeros (Bot)
        y_bot = np.zeros((81, 9), dtype=np.float32)
        return x, y_bot, False

    raise ValueError(f"Unknown sample_type: {sample_type}")


def _augment(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    cfg: SudokuExtremeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply domain-agnostic augmentations to [81, 9] arrays.

    Only uses transforms that would work for any grid problem (e.g. ARC-AGI):
    digit permutation, rotation, and mirror.
    """
    if cfg.augment_digit_perm:
        perm = rng.permutation(9)
        x = x[:, perm]
        y = y[:, perm]

    if not cfg.augment_dihedral:
        return x, y

    # Reshape to 9x9 grid for spatial transforms
    x = x.reshape(9, 9, 9)
    y = y.reshape(9, 9, 9)

    # Sample uniformly from D4: 8 elements = 4 rotations × {identity, reflection}
    k = int(rng.integers(0, 4))  # rotation: 0/90/180/270
    reflect = bool(rng.integers(0, 2))  # whether to reflect

    if k > 0:
        x = np.rot90(x, k=k, axes=(0, 1)).copy()
        y = np.rot90(y, k=k, axes=(0, 1)).copy()
    if reflect:
        x = np.flip(x, axis=1).copy()  # reflect across vertical axis
        y = np.flip(y, axis=1).copy()

    return x.reshape(81, 9), y.reshape(81, 9)


def select_pool(n_total: int, cfg: SudokuExtremeConfig) -> tuple[np.ndarray, np.random.Generator]:
    """Absolute indices a dataset with `cfg` trains on, and the RNG it continues with.

    Applies the validation holdout, then the seeded `n_puzzles` subset.
    Baselines use it to train on the same puzzles as LDT.
    """
    pool = np.arange(n_total)
    if cfg.split == "train" and cfg.exclude_val:
        val = index_lists.load("sudoku_val", split_size=n_total).indices
        pool = np.setdiff1d(pool, val, assume_unique=True)
    rng = np.random.default_rng(cfg.seed)
    if cfg.n_puzzles is not None and cfg.n_puzzles < len(pool):
        pool = np.sort(rng.choice(pool, size=cfg.n_puzzles, replace=False))
    return pool, rng


class SudokuExtremeDataset:
    """Streaming on-demand dataset for sudoku powerset training.

    Generates samples on the fly with configurable hint types and augmentations.
    Background thread prefetches batches for GPU transfer.
    Fully deterministic given a seed.
    """

    def __init__(self, cfg: SudokuExtremeConfig):
        self.cfg = cfg

        # Load base puzzles
        cache_path = _download_and_cache(cfg.cache_dir, cfg.split)
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.questions = data["x"].numpy()  # [N, 81, 9] uint8
        self.answers = data["y"].numpy()    # [N, 81, 9] uint8
        n_total = self.questions.shape[0]

        pool, self.rng = select_pool(n_total, cfg)
        if len(pool) < n_total:
            self.questions = self.questions[pool]
            self.answers = self.answers[pool]
        self.pool_indices = pool        # absolute dataset indices, after holdout and subsetting

        self.n_puzzles = self.questions.shape[0]

        # Sample type weights
        weights = np.array([cfg.zero_hint_weight, cfg.correct_hint_weight, cfg.error_hint_weight])
        self.type_probs = weights / weights.sum()
        self.type_names = ["zero_hints", "correct_hints", "error_hints"]

        # Iteration state
        self._order = self.rng.permutation(self.n_puzzles)
        self._pos = 0

        # Prefetch queue
        self._queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
            maxsize=cfg.prefetch_batches
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _next_sample(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Generate the next sample from the current puzzle order."""
        if self._pos >= self.n_puzzles:
            self._order = self.rng.permutation(self.n_puzzles)
            self._pos = 0

        idx = self._order[self._pos]
        self._pos += 1

        q = self.questions[idx].copy()
        a = self.answers[idx].copy()

        sample_type = self.rng.choice(self.type_names, p=self.type_probs)
        x, y, is_sat = _make_sample(q, a, self.rng, sample_type, self.cfg)
        x, y = _augment(x, y, self.rng, self.cfg)
        return x, y, is_sat

    def _prefetch_loop(self):
        """Background thread that fills the batch queue."""
        while not self._stop.is_set():
            batch_x, batch_y, batch_sat = [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, is_sat = self._next_sample()
                batch_x.append(x)
                batch_y.append(y)
                batch_sat.append(is_sat)

            tx = torch.from_numpy(np.stack(batch_x))
            ty = torch.from_numpy(np.stack(batch_y))
            tsat = torch.tensor(batch_sat, dtype=torch.bool)

            try:
                self._queue.put((tx, ty, tsat), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next prefetched batch.

        Returns:
            (x, y, is_sat) where x: [B, 81, 9], y: [B, 81, 9], is_sat: [B]
        """
        return self._queue.get()

    def close(self):
        """Stop the prefetch thread."""
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __del__(self):
        self.close()
