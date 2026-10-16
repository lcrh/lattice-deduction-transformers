"""Data pipeline for snowflake sudoku under a shared covering lattice.

Every puzzle (regardless of n) is embedded into a fixed 15×10 grid of cells.
Each cell has a one-hot powerset over 6 digits. Cells not present in a
given puzzle are zeroed and masked. The vocabulary is fixed at V=6.

The covering grid is derived from the union of all hex positions that
appear in the snowflake-sudoku topology module for n=1..19. Each
hexagonal cell's `(q, r, direction)` maps to a fixed `(row, col)` in
the covering grid. This is enough slots to embed every puzzle up to
n=19; the generator translates its grown layouts into the same box. The
representation is shared across orders, which is what makes
order-transfer experiments possible.

A `SnowflakeDataset` generates samples with the same three sample types
as the Sudoku dataset (zero_hints, correct_hints, error_hints) so the
CLS conflict head gets UNSAT training signal from corrupted givens.
Optional data-side augmentation: random whole-hex translation within the
covering grid (`translate_aug`). Optional inputs: randomized multi-hot
constraint-group channels (`constraint_group_embed`).

Splits are parquet files `snowflake_<split>.parquet` in the data directory
(see `split_file`).
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

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
BLANK = 7       # puzzle value of an empty cell
# Order 19 has at most 19 hex groups + 24 meeting-point groups. Group IDs are
# sampled from this whole range for every order, so every channel is used.
MAX_CONSTRAINT_GROUPS = 43


def split_file(data_dir: str, split: str) -> Path:
    """Parquet file of `split` ("train", "val" or "test")."""
    return Path(data_dir) / f"snowflake_{split}.parquet"


def cell_index(cp: dict) -> int:
    """Covering-grid index of a topology `cell_positions` entry."""
    return cell_to_grid_idx(cp["q"], cp["r"], cp["direction"])


def cell_to_grid_idx(q: int, r: int, direction: str) -> int:
    dr, dc = DIR_OFFSET[direction]
    row = 3 * (r - R_MIN) + dr
    col = 2 * (q - Q_MIN) + dc
    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        raise ValueError(f"cell (q={q}, r={r}, {direction}) lies outside the "
                         f"{GRID_ROWS}x{GRID_COLS} covering grid")
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
        idx = cell_index(cp)
        mask[idx] = True
        # Ground-truth one-hot (digits 1..6 → indices 0..5).
        y[idx, solution[cell_id] - 1] = 1.0
        # Input powerset: given → one-hot; blank (value=7) → all-ones.
        val = puzzle[cell_id]
        if val == BLANK:
            x[idx, :] = 1.0
        else:
            x[idx, val - 1] = 1.0
    return x, y, mask


def constraint_group_features(puzzle_rec: dict, rng: np.random.Generator) -> np.ndarray:
    """[SEQ_LEN, MAX_CONSTRAINT_GROUPS] multi-hot constraint membership.

    Each constraint group gets a distinct ID drawn at random from the whole
    ID range, resampled on every call, so the model can use shared membership
    but cannot tie an ID to a location.
    """
    topology = puzzle_rec["topology"]
    constraints = topology["constraints"]
    if len(constraints) > MAX_CONSTRAINT_GROUPS:
        raise ValueError(f"puzzle has {len(constraints)} constraint groups, "
                         f"more than {MAX_CONSTRAINT_GROUPS}")
    ids = rng.choice(MAX_CONSTRAINT_GROUPS, size=len(constraints), replace=False)
    features = np.zeros((SEQ_LEN, MAX_CONSTRAINT_GROUPS), dtype=np.float32)
    positions = topology["cell_positions"]
    for group, group_id in zip(constraints, ids):
        for cell_id in group["cells"]:
            features[cell_index(positions[str(cell_id)]), group_id] = 1.0
    return features


def random_translate(arrays: list[np.ndarray], mask: np.ndarray,
                     rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Shift a puzzle by a random whole number of hexes within the covering grid.

    A hex is a 3 x 2 block of slots, so the shift keeps which cells share
    constraints. `arrays` ([SEQ_LEN, ...]) move together with `mask`.
    """
    occupied = np.where(mask)[0]
    if occupied.size == 0:
        return arrays, mask
    rows, cols = occupied // GRID_COLS, occupied % GRID_COLS
    dr = int(rng.integers(-(rows.min() // 3), (GRID_ROWS - 1 - rows.max()) // 3 + 1))
    dq = int(rng.integers(-(cols.min() // 2), (GRID_COLS - 1 - cols.max()) // 2 + 1))
    if dr == 0 and dq == 0:
        return arrays, mask
    new_idx = (rows + 3 * dr) * GRID_COLS + (cols + 2 * dq)
    moved = []
    for a in arrays:
        out = np.zeros_like(a)
        out[new_idx] = a[occupied]
        moved.append(out)
    new_mask = np.zeros_like(mask)
    new_mask[new_idx] = True
    return moved, new_mask


def puzzle_orders(puzzles: list[dict]) -> np.ndarray:
    return np.array([int(p["n"]) for p in puzzles], dtype=np.int64)


def select_pool(puzzles: list[dict], cfg: "SnowflakeConfig") -> np.ndarray:
    """Row indices a dataset with `cfg` trains on: `train_orders`, then a seeded `n_puzzles` subset."""
    orders = puzzle_orders(puzzles)
    rng = np.random.default_rng(cfg.seed)
    pool = np.arange(len(puzzles))
    if cfg.train_orders is not None:
        pool = pool[np.isin(orders, cfg.train_orders)]
    if cfg.n_puzzles is not None and cfg.n_puzzles < len(pool):
        pool = pool[rng.choice(len(pool), cfg.n_puzzles, replace=False)]
    if len(pool) == 0:
        raise ValueError(f"no puzzles match train_orders={cfg.train_orders}")
    return pool


# -----------------------------------------------------------------------------
# Dataset with on-the-fly sample generation.


@dataclass
class SnowflakeConfig:
    data_dir: str = "data"
    n_puzzles: int | None = None        # subset size, None = all
    train_orders: list[int] | None = None   # keep only these orders (None = all)
    translate_aug: bool = False         # random whole-hex translation per sample
    constraint_group_embed: bool = False    # append constraint-group channels
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
    """Apply a sample type to x in place; returns (is_sat, target)."""
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
        return False, np.zeros_like(y)      # unsatisfiable: empty target

    raise ValueError(f"unknown sample_type {sample_type!r}")


def load_puzzles(path: str | Path) -> list[dict]:
    """Rows of a generator parquet file, with the JSON topology decoded."""
    import pyarrow.parquet as pq
    rows = pq.read_table(str(path)).to_pylist()
    for r in rows:
        r["topology"] = json.loads(r["topology"])
    return rows


def load_states(puzzles: list[dict], indices, rng: np.random.Generator | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Zero-hint (x, y, in_puzzle_mask, groups) tensors for `puzzles[i]`, i in `indices`.

    `groups` (constraint-group channels) is built only when `rng` is given.
    """
    xs, ys, ms = zip(*(puzzle_to_state(puzzles[int(i)]) for i in indices))
    groups = None
    if rng is not None:
        groups = torch.from_numpy(np.stack(
            [constraint_group_features(puzzles[int(i)], rng) for i in indices]))
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ms)), groups)


class SnowflakeDataset:
    def __init__(self, cfg: SnowflakeConfig):
        self.cfg = cfg
        puzzles = load_puzzles(split_file(cfg.data_dir, "train"))
        self.pool_indices = select_pool(puzzles, cfg)
        self.puzzles = [puzzles[i] for i in self.pool_indices]
        self.n_puzzles = len(self.puzzles)

        self.rng = np.random.default_rng(cfg.seed)
        weights = np.array([cfg.zero_hint_weight, cfg.correct_hint_weight, cfg.error_hint_weight])
        self.type_probs = weights / weights.sum()
        self.type_names = ["zero_hints", "correct_hints", "error_hints"]

        self._order = self.rng.permutation(self.n_puzzles)
        self._pos = 0

        self._queue: queue.Queue = queue.Queue(maxsize=cfg.prefetch_batches)
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
        arrays = [x, y]
        if self.cfg.constraint_group_embed:
            arrays.append(constraint_group_features(rec, self.rng))
        if self.cfg.translate_aug:
            arrays, mask = random_translate(arrays, mask, self.rng)
        x, y = arrays[0], arrays[1]
        groups = arrays[2] if self.cfg.constraint_group_embed else None
        stype = self.rng.choice(self.type_names, p=self.type_probs)
        is_sat, y = _apply_sample_type(x, y, mask, stype, self.cfg, self.rng)
        return x, y, mask, is_sat, groups

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, bm, bs, bg = [], [], [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, mask, is_sat, groups = self._next_sample()
                for buf, v in zip((bx, by, bm, bs, bg), (x, y, mask, is_sat, groups)):
                    buf.append(v)
            tx = torch.from_numpy(np.stack(bx))
            ty = torch.from_numpy(np.stack(by))
            tm = torch.from_numpy(np.stack(bm))
            ts = torch.tensor(bs, dtype=torch.bool)
            tg = torch.from_numpy(np.stack(bg)) if self.cfg.constraint_group_embed else None
            try:
                self._queue.put((tx, ty, tm, ts, tg), timeout=1.0)
            except queue.Full:
                if self._stop.is_set():
                    break

    def next_batch(self):
        """(x, y, in_puzzle_mask, is_sat, groups); groups is None unless constraint_group_embed."""
        return self._queue.get()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __del__(self):
        self.close()
