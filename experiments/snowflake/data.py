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


# ORDER-INDEPENDENCE (verified for E4 order-transfer OOD): the covering-grid
# embedding above is a pure function of the per-cell (q, r, direction) hex
# coordinates and the FIXED grid extents (Q_MIN/Q_MAX/R_MIN/R_MAX, i.e.
# GRID_ROWS=15, GRID_COLS=10). There is NO dependency on the puzzle order `n`:
# `cell_to_grid_idx` never reads `n`, and `puzzle_to_state` derives every slot
# purely from `puzzle_rec["topology"]["cell_positions"]` (the concrete cells of
# the given puzzle) — no per-order normalization, rescaling, or recentering.
# A cell at coordinate (q, r, dir) lands at the same grid slot regardless of the
# order of the puzzle it belongs to. This is exactly what makes training on
# small orders and evaluating on larger orders well-posed: a shared, order-
# agnostic slot layout. (The only order-dependence is WHICH slots are occupied,
# which is the positional confound that translation aug / RoPE address.)


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
# Translation augmentation (E4 positional-confound mitigation).


def random_translate(
    x: np.ndarray,      # [SEQ_LEN, VOCAB]
    y: np.ndarray,      # [SEQ_LEN, VOCAB]
    mask: np.ndarray,   # [SEQ_LEN] bool
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly translate a puzzle within the 15x10 covering grid.

    Design.  Every hex cell (q, r, direction) maps to a grid slot via
    `cell_to_grid_idx`:  row = 3*(r - R_MIN) + dr,  col = 2*(q - Q_MIN) + dc,
    where (dr, dc) ranges over the 6 DIR_OFFSET sub-block entries (dr in {0,1,2},
    dc in {0,1}). A full hex cell therefore occupies a 3-row x 2-col sub-block.
    To translate the puzzle by an integer hex offset (dq, dr_hex) WITHOUT
    breaking that sub-block structure, we shift the grid row of every occupied
    slot by  3*dr_hex  and the grid col by  2*dq. Because the shift is a whole
    multiple of the sub-block size, the 6-direction internal layout of each hex
    cell is preserved exactly, and the transform is a pure permutation of grid
    slots applied identically to x, y and mask (relabelling positions, never
    touching the vocab channels).

    Bounds check.  We compute the occupied bounding box (min/max row and col
    over `mask`), then sample the shift so that after shifting no occupied slot
    leaves the [0, GRID_ROWS) x [0, GRID_COLS) grid:
        row shift srow = 3*dr_hex in [-min_row, GRID_ROWS-1 - max_row]  (step 3)
        col shift scol = 2*dq     in [-min_col, GRID_COLS-1 - max_col]  (step 2)
    The legal integer range for dr_hex / dq is derived by dividing those slot
    bounds by the step and flooring/ceiling appropriately. If the occupied box
    already spans the grid in a dimension, that dimension's shift is 0.

    Soundness.  The AllDifferent constraint groups of a Snowflake puzzle are a
    function of the visible hex geometry (which cells share a line/ring), which
    a rigid translation preserves — it only relabels covering-grid positions.
    So the translated (x, y, mask) is a valid, equivalent puzzle with the same
    solution structure; only the absolute grid positions change, which is
    exactly the invariance we want the positional embedding to learn.
    """
    occ = np.where(mask)[0]
    if occ.size == 0:
        return x, y, mask
    rows = occ // GRID_COLS
    cols = occ % GRID_COLS
    min_row, max_row = int(rows.min()), int(rows.max())
    min_col, max_col = int(cols.min()), int(cols.max())

    # Legal hex-cell shift ranges (units of full hex cells: 3 rows / 2 cols).
    # srow = 3*dr_hex must satisfy -min_row <= srow <= (GRID_ROWS-1) - max_row.
    dr_lo = -(min_row // 3)                                  # ceil(-min_row / 3)
    dr_hi = (GRID_ROWS - 1 - max_row) // 3                   # floor(slack / 3)
    dq_lo = -(min_col // 2)
    dq_hi = (GRID_COLS - 1 - max_col) // 2

    dr_hex = int(rng.integers(dr_lo, dr_hi + 1)) if dr_hi >= dr_lo else 0
    dq = int(rng.integers(dq_lo, dq_hi + 1)) if dq_hi >= dq_lo else 0
    if dr_hex == 0 and dq == 0:
        return x, y, mask

    srow = 3 * dr_hex
    scol = 2 * dq
    new_rows = rows + srow
    new_cols = cols + scol
    new_idx = new_rows * GRID_COLS + new_cols

    x_out = np.zeros_like(x)
    y_out = np.zeros_like(y)
    mask_out = np.zeros_like(mask)
    x_out[new_idx] = x[occ]
    y_out[new_idx] = y[occ]
    mask_out[new_idx] = True
    return x_out, y_out, mask_out


# -----------------------------------------------------------------------------
# Dataset with on-the-fly sample generation.


@dataclass
class SnowflakeConfig:
    data_path: str = "data/snowflake_train.parquet"
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
    # E4 order-transfer OOD knobs (all default-off / default-preserving):
    #   orders: if not None, keep only puzzles whose `n` is in this list
    #           (applied BEFORE the n_puzzles subset selection).
    #   translate_aug: if True, apply a random covering-grid translation
    #           jointly to (x, y, in_puzzle_mask) per sample (see
    #           `random_translate`). Default off — a plain run is unchanged.
    #   return_orders: if True, `next_batch()` returns a 5-tuple
    #           (x, y, mask, sat, orders) where `orders` is a [B] long tensor
    #           of each sample's puzzle order `n`; default False keeps the
    #           existing 4-tuple contract for all current callers.
    orders: list[int] | None = None
    translate_aug: bool = False
    return_orders: bool = False


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


class SnowflakeDataset:
    def __init__(self, cfg: SnowflakeConfig):
        self.cfg = cfg
        self.puzzles = _load_puzzles(cfg.data_path)
        # E4 order filtering: keep only puzzles whose order `n` is requested.
        # Applied BEFORE the n_puzzles subset selection so the subset is drawn
        # from the filtered pool. Applies to both train and eval loaders.
        if cfg.orders is not None:
            allowed = set(int(o) for o in cfg.orders)
            self.puzzles = [r for r in self.puzzles if int(r["n"]) in allowed]
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
        # E4 translation aug: shift the whole occupied region within the grid.
        # Applied BEFORE sample-type corruption so corruption operates on the
        # translated (blank) cells; the transform is a pure position relabelling
        # and commutes with sample-type logic (which is per-cell, position-blind).
        if self.cfg.translate_aug:
            x, y, mask = random_translate(x, y, mask, self.rng)
        stype = self.rng.choice(self.type_names, p=self.type_probs)
        is_sat, y = _apply_sample_type(x, y, mask, stype, self.cfg, self.rng)
        order = int(rec["n"])
        return x, y, mask, is_sat, order

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, bm, bs, bo = [], [], [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, mask, is_sat, order = self._next_sample()
                bx.append(x); by.append(y); bm.append(mask); bs.append(is_sat)
                bo.append(order)
            tx = torch.from_numpy(np.stack(bx))
            ty = torch.from_numpy(np.stack(by))
            tm = torch.from_numpy(np.stack(bm))
            ts = torch.tensor(bs, dtype=torch.bool)
            to = torch.tensor(bo, dtype=torch.long)
            try:
                self._queue.put((tx, ty, tm, ts, to), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def next_batch(self):
        """Return one prefetched batch.

        By default returns the 4-tuple (x, y, mask, sat) that all existing
        callers unpack. When `cfg.return_orders` is True, returns the 5-tuple
        (x, y, mask, sat, orders) with `orders` a [B] long tensor of per-sample
        puzzle order `n` (E4 per-order eval breakdown).
        """
        tx, ty, tm, ts, to = self._queue.get()
        if self.cfg.return_orders:
            return tx, ty, tm, ts, to
        return tx, ty, tm, ts

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __del__(self):
        self.close()
