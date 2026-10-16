"""Data pipeline for snowflake sudoku under a shared covering lattice.

Every puzzle (regardless of n) is embedded into a fixed 15×10 grid of cells.
Each cell has a one-hot powerset over 6 digits. Cells not present in a
given puzzle are zeroed and masked. The vocabulary is fixed at V=6.

The covering grid is derived from the union of all hex positions that
appear in the snowflake-sudoku topology module for n=1..19. Each
hexagonal cell's `(q, r, direction)` maps to a fixed `(row, col)` in
the covering grid. This is enough slots to embed every puzzle up to
n=19; training uses only the 30k generated for n=4..8 but the
representation is shared.

A `SnowflakeDataset` generates samples with the same three sample types
as the Sudoku dataset (zero_hints, correct_hints, error_hints) so the
CLS conflict head gets UNSAT training signal from corrupted givens.
No augmentation.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Covering lattice

# (q, r) ∈ [-2, 2]² and 6 directions → 15 × 10 = 150 slots.
Q_MIN, Q_MAX = -2, 2
R_MIN, R_MAX = -2, 2
DIR_OFFSET = {
    "NW": (0, 0), "NE": (0, 1),
    "W":  (1, 0), "E":  (1, 1),
    "SW": (2, 0), "SE": (2, 1),
}
GRID_ROWS = 3 * (R_MAX - R_MIN + 1)   # 15
GRID_COLS = 2 * (Q_MAX - Q_MIN + 1)   # 10
SEQ_LEN = GRID_ROWS * GRID_COLS       # 150
VOCAB = 6


def cell_to_grid_idx(q: int, r: int, direction: str) -> int:
    dr, dc = DIR_OFFSET[direction]
    row = 3 * (r - R_MIN) + dr
    col = 2 * (q - Q_MIN) + dc
    return row * GRID_COLS + col


def puzzle_to_state(puzzle_rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a snowflake puzzle record into covering-grid tensors.

    Returns:
      x:  [SEQ_LEN, VOCAB] powerset state — all-ones for blanks, one-hot for
          givens, all-zeros for cells not in this puzzle.
      y:  [SEQ_LEN, VOCAB] one-hot ground-truth solution for in-puzzle cells,
          all-zeros elsewhere.
      in_puzzle_mask: [SEQ_LEN] bool, True for cells in this puzzle.
    """
    x = np.zeros((SEQ_LEN, VOCAB), dtype=np.float32)
    y = np.zeros((SEQ_LEN, VOCAB), dtype=np.float32)
    mask = np.zeros((SEQ_LEN,), dtype=bool)

    cell_positions = puzzle_rec["topology"]["cell_positions"]
    puzzle = puzzle_rec["puzzle"]
    solution = puzzle_rec["solution"]

    for cell_id_str, cp in cell_positions.items():
        cell_id = int(cell_id_str)
        idx = cell_to_grid_idx(cp["q"], cp["r"], cp["direction"])
        mask[idx] = True
        # Ground-truth one-hot (digits 1..6 → indices 0..5).
        y[idx, solution[cell_id] - 1] = 1.0
        # Input powerset: given → one-hot; blank (value=7) → all-ones.
        val = puzzle[cell_id]
        if val == 7:
            x[idx, :] = 1.0
        else:
            x[idx, val - 1] = 1.0
    return x, y, mask


# -----------------------------------------------------------------------------
# Dataset with on-the-fly sample generation.


@dataclass
class SnowflakeConfig:
    data_dir: str = "data"
    data_path: str = "snowflake_train.parquet"   # relative to data_dir (or absolute)
    n_puzzles: int | None = None        # subset size, None = all
    seed: int = 42
    batch_size: int = 512
    # Mirror the sudoku-extreme sample-type weights so CLS gets UNSAT signal.
    zero_hint_weight: float = 0.20
    correct_hint_weight: float = 0.55
    error_hint_weight: float = 0.25
    correct_fill_range: tuple[float, float] = (0.0, 1.0)
    error_fill_range: tuple[float, float] = (0.1, 1.0)
    error_rate_range: tuple[float, float] = (0.01, 0.30)
    prefetch_batches: int = 2


def _apply_sample_type(x, y, mask, sample_type, cfg, rng):
    """Variant of the sudoku-extreme _make_sample adapted for snowflake.

    Operates in-place on x, y. Returns is_sat (True/False) and the (possibly
    modified) y target.
    """
    if sample_type == "zero_hints":
        return True, y

    # Identify blank in-puzzle cells: in mask AND sum(x)==VOCAB (all-ones row).
    blanks = mask & (x.sum(axis=1) > 1.5)
    blank_indices = np.where(blanks)[0]
    if blank_indices.size == 0:
        return True, y

    if sample_type == "correct_hints":
        lo, hi = cfg.correct_fill_range
        fill = rng.uniform(lo, hi)
        n_fill = int(fill * blank_indices.size)
        if n_fill > 0:
            to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
            x[to_fill] = y[to_fill]
        return True, y

    if sample_type == "error_hints":
        lo, hi = cfg.error_fill_range
        fill = rng.uniform(lo, hi)
        n_fill = max(1, int(fill * blank_indices.size))
        to_fill = rng.choice(blank_indices, size=n_fill, replace=False)
        x[to_fill] = y[to_fill]
        elo, ehi = cfg.error_rate_range
        n_corrupt = max(1, int(rng.uniform(elo, ehi) * n_fill))
        to_corrupt = rng.choice(to_fill, size=n_corrupt, replace=False)
        for idx in to_corrupt:
            correct_digit = int(y[idx].argmax())
            wrong_digit = rng.choice([d for d in range(VOCAB) if d != correct_digit])
            x[idx] = 0.0
            x[idx, wrong_digit] = 1.0
        # Target becomes Bot (all zeros) for the IN-PUZZLE cells.
        y_bot = np.zeros_like(y)
        # Out-of-puzzle cells stay zero (as they are). No change needed.
        return False, y_bot

    raise ValueError(f"unknown sample_type {sample_type!r}")


def _load_puzzles(data_path: str) -> list[dict]:
    """Load puzzle records from a JSON file or a parquet file/dir.

    Parquet rows must have columns (id, n, code, puzzle, solution, givens,
    topology) where `topology` is a JSON-encoded string of the nested
    topology dict. This matches `experiments/snowflake/gen_data.py`'s
    output. Plain JSON lists with the same row schema are also supported.
    """
    p = Path(data_path)
    if p.suffix == ".parquet" or p.is_dir():
        import pyarrow.parquet as pq
        # Single-file or directory-of-shards.
        if p.is_dir():
            files = sorted(p.glob("*.parquet"))
            tables = [pq.read_table(str(f)) for f in files]
            import pyarrow as pa
            table = pa.concat_tables(tables)
        else:
            table = pq.read_table(str(p))
        rows = table.to_pylist()
        # Decode the JSON-string topology column.
        for r in rows:
            r["topology"] = json.loads(r["topology"])
        return rows
    with open(data_path) as f:
        return json.load(f)


def load_states(puzzles: list[dict], indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, y, in_puzzle_mask) for `puzzles[i]`, i in `indices`, as zero-hint
    covering-grid tensors."""
    xs, ys, ms = zip(*(puzzle_to_state(puzzles[int(i)]) for i in indices))
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ms)))


class SnowflakeDataset:
    def __init__(self, cfg: SnowflakeConfig):
        self.cfg = cfg
        self.puzzles = _load_puzzles(str(Path(cfg.data_dir) / cfg.data_path))
        if cfg.n_puzzles is not None and cfg.n_puzzles < len(self.puzzles):
            rng_init = np.random.default_rng(cfg.seed)
            idx = rng_init.choice(len(self.puzzles), cfg.n_puzzles, replace=False)
            self.puzzles = [self.puzzles[i] for i in idx]
        self.n_puzzles = len(self.puzzles)

        self.rng = np.random.default_rng(cfg.seed)
        weights = np.array([cfg.zero_hint_weight, cfg.correct_hint_weight, cfg.error_hint_weight])
        self.type_probs = weights / weights.sum()
        self.type_names = ["zero_hints", "correct_hints", "error_hints"]

        self._order = self.rng.permutation(self.n_puzzles)
        self._pos = 0

        self._queue: Queue = Queue(maxsize=cfg.prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _next_sample(self):
        if self._pos >= self.n_puzzles:
            self._order = self.rng.permutation(self.n_puzzles)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        rec = self.puzzles[idx]
        x, y, mask = puzzle_to_state(rec)
        stype = self.rng.choice(self.type_names, p=self.type_probs)
        is_sat, y = _apply_sample_type(x, y, mask, stype, self.cfg, self.rng)
        return x, y, mask, is_sat

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, bm, bs = [], [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, mask, is_sat = self._next_sample()
                bx.append(x); by.append(y); bm.append(mask); bs.append(is_sat)
            tx = torch.from_numpy(np.stack(bx))
            ty = torch.from_numpy(np.stack(by))
            tm = torch.from_numpy(np.stack(bm))
            ts = torch.tensor(bs, dtype=torch.bool)
            try:
                self._queue.put((tx, ty, tm, ts), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def next_batch(self):
        return self._queue.get()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __del__(self):
        self.close()
