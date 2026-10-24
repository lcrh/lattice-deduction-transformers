"""Streaming dataset for maze-30x30-hard powerset training.

Downloads sapientinc/maze-30x30-hard-1k from HuggingFace, caches as a .pt
file, and serves on-the-fly augmented samples for path-prediction training.

Format:
- Each puzzle is a 30x30 grid with cells in {wall '#', free ' ', start 'S',
  goal 'G'} for the question and {wall, free, start, goal, path 'o'} for the
  answer. Path cells are otherwise-free cells the solver should mark.
- We encode each cell as a 5-channel one-hot over (wall, free, start, goal,
  path).
- The input `x` is the powerset/lattice initial state: walls/start/goal are
  singletons (already determined); free cells have BOTH `free` and `path`
  channels alive (the model must decide which). `y` is the singleton ground
  truth.
- Note on S/G: in the answer encoding, S and G stay labeled as start/goal
  (path cells are 'o'), so we treat them as singletons in `x` rather than
  letting them carry an extra `path` candidate. The solver therefore deduces
  free-vs-path only on cells that were originally blank space.
- Domain-agnostic augmentations: D4 (rotations + reflections — applied
  consistently to x and y, including the start/goal channels which
  transform with the spatial axes) and an optional S/G channel swap (mazes
  are spatially bidirectional, so swapping the start/goal labels in both x
  and y is a valid relabeling that preserves the path geometry).
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np
import torch


GRID_H = 30
GRID_W = 30
N_CELLS = GRID_H * GRID_W

# Channel ordering — kept consistent everywhere.
CH_WALL = 0
CH_FREE = 1
CH_START = 2
CH_GOAL = 3
CH_PATH = 4
N_CHANNELS = 5

# Map question characters to a singleton channel; free has two alive channels
# in the powerset input (free OR path), but the question itself encodes a
# blank space, so the singleton encoding here is just "free."
_QCHAR_TO_CH = {
    ord('#'): CH_WALL,
    ord(' '): CH_FREE,
    ord('S'): CH_START,
    ord('G'): CH_GOAL,
}
# The answer can also have 'o' marking a path cell.
_ACHAR_TO_CH = {
    ord('#'): CH_WALL,
    ord(' '): CH_FREE,
    ord('S'): CH_START,
    ord('G'): CH_GOAL,
    ord('o'): CH_PATH,
}


@dataclass
class MazeHardConfig:
    cache_dir: str = "data"
    split: str = "train"  # "train" or "test"
    n_puzzles: int | None = None
    seed: int = 42

    # Domain-agnostic augmentations.
    augment_dihedral: bool = True
    augment_swap_endpoints: bool = True

    # Diagnostic mode: replace the actual maze + path with a wall-less grid
    # whose GT is the Bresenham straight line from S to G. Same shape and
    # encoding as the real puzzle, but the model only needs to identify
    # which cells lie on the straight line between the two singleton
    # endpoints. Useful for testing whether the model can use S/G info at
    # all.
    simplify_to_straight_line: bool = False

    prefetch_batches: int = 2
    batch_size: int = 64


def _download_and_cache(cache_dir: str, split: str = "train") -> Path:
    """Download maze-30x30-hard-1k and cache as .pt with one-hot tensors."""
    cache_path = Path(cache_dir) / f"maze_hard_{split}.pt"
    if cache_path.exists():
        return cache_path

    from huggingface_hub import hf_hub_download

    print(f"Downloading sapientinc/maze-30x30-hard-1k {split}.csv …", flush=True)
    csv_path = hf_hub_download(
        repo_id="sapientinc/maze-30x30-hard-1k",
        filename=f"{split}.csv",
        repo_type="dataset",
    )

    questions = []
    answers = []
    n_dropped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # IMPORTANT: do NOT strip whitespace. Free cells in this dataset
            # are encoded as ' ', so leading/trailing spaces are real cells.
            # `.strip()` or `.lstrip(' ')` would silently truncate rows and
            # drop them at the length check below. Take the row as-is; the
            # post-lookup unknown-char check raises on malformed input.
            q = row["question"]
            a = row["answer"]
            if len(q) != N_CELLS or len(a) != N_CELLS:
                n_dropped += 1
                continue
            questions.append(q)
            answers.append(a)
    n = len(questions)
    if n_dropped > 0:
        print(f"  dropped {n_dropped} malformed rows (length != {N_CELLS})", flush=True)
    print(f"  {n} mazes. Encoding …", flush=True)

    # Build a flat byte→channel lookup table (vectorized encode). Unknown
    # bytes get a sentinel (255) so we can detect them post-lookup rather
    # than silently encoding them as free.
    UNKNOWN = np.uint8(255)
    q_lookup = np.full(256, UNKNOWN, dtype=np.uint8)
    a_lookup = np.full(256, UNKNOWN, dtype=np.uint8)
    for c, ch in _QCHAR_TO_CH.items():
        q_lookup[c] = ch
    for c, ch in _ACHAR_TO_CH.items():
        a_lookup[c] = ch

    q_arr = np.frombuffer("".join(questions).encode("ascii"), dtype=np.uint8).reshape(n, N_CELLS)
    a_arr = np.frombuffer("".join(answers).encode("ascii"), dtype=np.uint8).reshape(n, N_CELLS)
    q_idx = q_lookup[q_arr]  # [n, 900] in {0..4} or 255 for unknown
    a_idx = a_lookup[a_arr]  # [n, 900] in {0..4} or 255 for unknown
    if (q_idx == UNKNOWN).any() or (a_idx == UNKNOWN).any():
        bad_q = sorted({chr(c) for c in q_arr[q_idx == UNKNOWN].tolist()})
        bad_a = sorted({chr(c) for c in a_arr[a_idx == UNKNOWN].tolist()})
        raise ValueError(
            f"Unknown characters in maze CSV: question chars {bad_q}, "
            f"answer chars {bad_a}. Expected only {sorted(_ACHAR_TO_CH.keys())}."
        )

    one_hot = np.eye(N_CHANNELS, dtype=np.uint8)
    q_onehot = one_hot[q_idx]  # [n, 900, 5]
    a_onehot = one_hot[a_idx]  # [n, 900, 5]

    # Powerset encoding for x: free cells have BOTH free and path alive.
    # Walls/start/goal stay singleton.
    x = q_onehot.copy()
    free_mask = q_idx == CH_FREE  # [n, 900]
    x[free_mask, CH_PATH] = 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(a_onehot),
            "n": n,
            "grid_h": GRID_H,
            "grid_w": GRID_W,
            "n_channels": N_CHANNELS,
        },
        cache_path,
    )
    print(f"  cached to {cache_path}", flush=True)
    return cache_path


def _augment(
    x: np.ndarray,  # [900, 5]
    y: np.ndarray,  # [900, 5]
    rng: np.random.Generator,
    cfg: MazeHardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """D4 spatial augmentations + optional S/G channel swap."""
    if cfg.augment_swap_endpoints and bool(rng.integers(0, 2)):
        # Permute the start <-> goal channels in both x and y. This relabels
        # a maze "from S to G" as "from G to S"; the path cells are the same
        # geometrically, so y stays valid.
        x = x.copy()
        y = y.copy()
        x[:, [CH_START, CH_GOAL]] = x[:, [CH_GOAL, CH_START]]
        y[:, [CH_START, CH_GOAL]] = y[:, [CH_GOAL, CH_START]]

    if not cfg.augment_dihedral:
        return x, y

    x = x.reshape(GRID_H, GRID_W, N_CHANNELS)
    y = y.reshape(GRID_H, GRID_W, N_CHANNELS)

    k = int(rng.integers(0, 4))
    reflect = bool(rng.integers(0, 2))
    if k > 0:
        x = np.rot90(x, k=k, axes=(0, 1)).copy()
        y = np.rot90(y, k=k, axes=(0, 1)).copy()
    if reflect:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()

    return x.reshape(N_CELLS, N_CHANNELS), y.reshape(N_CELLS, N_CHANNELS)


def _to_straight_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace the maze with a wall-less puzzle whose target is a straight S-to-G line.

    Removes all walls (every non-S/G cell becomes a free-cell powerset state
    in x, i.e., both `free` and `path` alive). Computes the Bresenham-like
    straight-line cells between S and G via `np.linspace`-rounded
    interpolation, and marks those cells as `path` in y. All other free
    cells in y stay `free`. S and G remain singletons in both x and y.
    """
    x = x.reshape(GRID_H, GRID_W, N_CHANNELS).copy()
    y = y.reshape(GRID_H, GRID_W, N_CHANNELS).copy()

    s_pos = np.argwhere(y[:, :, CH_START] > 0.5)[0]
    g_pos = np.argwhere(y[:, :, CH_GOAL] > 0.5)[0]

    n_steps = max(abs(int(g_pos[0]) - int(s_pos[0])),
                  abs(int(g_pos[1]) - int(s_pos[1]))) + 1
    rs = np.linspace(s_pos[0], g_pos[0], n_steps).round().astype(int)
    cs = np.linspace(s_pos[1], g_pos[1], n_steps).round().astype(int)
    line_cells = set(zip(rs.tolist(), cs.tolist()))
    line_cells.discard((int(s_pos[0]), int(s_pos[1])))
    line_cells.discard((int(g_pos[0]), int(g_pos[1])))

    # New x: every cell is uncertain (free + path alive)
    x[:] = 0
    x[:, :, CH_FREE] = 1
    x[:, :, CH_PATH] = 1
    x[s_pos[0], s_pos[1]] = 0
    x[s_pos[0], s_pos[1], CH_START] = 1
    x[g_pos[0], g_pos[1]] = 0
    x[g_pos[0], g_pos[1], CH_GOAL] = 1

    # New y: free everywhere; path on line cells; S/G singletons
    y[:] = 0
    y[:, :, CH_FREE] = 1
    for r, c in line_cells:
        y[r, c] = 0
        y[r, c, CH_PATH] = 1
    y[s_pos[0], s_pos[1]] = 0
    y[s_pos[0], s_pos[1], CH_START] = 1
    y[g_pos[0], g_pos[1]] = 0
    y[g_pos[0], g_pos[1], CH_GOAL] = 1

    return x.reshape(N_CELLS, N_CHANNELS), y.reshape(N_CELLS, N_CHANNELS)


class MazeHardDataset:
    """Streaming dataset over maze-30x30-hard-1k.

    Generates samples on the fly with augmentations. Background thread
    prefetches batches. Deterministic given a seed.

    `next_batch()` returns (x, y, is_sat) matching the sudoku-extreme contract:
        x: [B, 900, 5] uint8 — powerset/lattice initial state
        y: [B, 900, 5] uint8 — singleton ground truth
        is_sat: [B] bool — always True (no UNSAT in this dataset)
    """

    def __init__(self, cfg: MazeHardConfig):
        self.cfg = cfg
        cache_path = _download_and_cache(cfg.cache_dir, cfg.split)
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.questions = data["x"].numpy()  # [N, 900, 5] uint8
        self.answers = data["y"].numpy()    # [N, 900, 5] uint8
        n_total = self.questions.shape[0]

        self.rng = np.random.default_rng(cfg.seed)
        if cfg.n_puzzles is not None and cfg.n_puzzles < n_total:
            indices = self.rng.choice(n_total, size=cfg.n_puzzles, replace=False)
            indices.sort()
            self.questions = self.questions[indices]
            self.answers = self.answers[indices]
        self.n_puzzles = self.questions.shape[0]

        self._order = self.rng.permutation(self.n_puzzles)
        self._pos = 0

        self._queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
            maxsize=cfg.prefetch_batches
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _next_sample(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._pos >= self.n_puzzles:
            self._order = self.rng.permutation(self.n_puzzles)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        # Cache is uint8 for size; PowersetModel's input_proj is nn.Linear so
        # we cast to float32 here (matches sudoku_extreme's _make_sample).
        x = self.questions[idx].astype(np.float32)
        y = self.answers[idx].astype(np.float32)
        x, y = _augment(x, y, self.rng, self.cfg)
        if self.cfg.simplify_to_straight_line:
            x, y = _to_straight_line(x, y)
        return x, y, True

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, bs = [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, is_sat = self._next_sample()
                bx.append(x); by.append(y); bs.append(is_sat)
            tx = torch.from_numpy(np.stack(bx))
            ty = torch.from_numpy(np.stack(by))
            tsat = torch.tensor(bs, dtype=torch.bool)
            try:
                self._queue.put((tx, ty, tsat), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._queue.get()

    def close(self):
        self._stop.set()
