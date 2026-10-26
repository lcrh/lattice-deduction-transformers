"""On-the-fly synthetic maze generator at arbitrary sizes.

Replicates the procedure from Lehnert et al. 2024 ("Beyond A*: Better
Planning with Transformers via Search Dynamics Bootstrapping", Searchformer
paper, arXiv:2402.14083), Appendix C:

  > Maze tasks were generated first by randomly selecting 30–50% of all cells
  > to be wall cells. Then a start and goal location was randomly selected
  > and A* was executed to obtain an optimal plan. If the plan had a length
  > of at least the maze's width or height [...] then the task was added
  > into the dataset.

We use 4-connectivity. Each generated puzzle satisfies:
- wall_frac ∈ [wall_frac_lo, wall_frac_hi]   (default 0.30–0.50)
- S and G are non-wall cells, S ≠ G
- A* finds a path of length ≥ max(h, w) cells
- (Optional) duplicate-rejection across the generation session

Encoding matches `data/maze_hard.py`: 5 channels (wall, free, start, goal,
path). x is the powerset/lattice initial state (free cells alive on both
free + path); y is the singleton GT (path cells marked).
"""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass
from queue import Queue

import numpy as np
import torch

# Re-export channel constants so callers don't have to import from maze_hard.
from ldt.data.maze_hard import (
    CH_WALL, CH_FREE, CH_START, CH_GOAL, CH_PATH, N_CHANNELS,
)


def _astar(walls: np.ndarray, s: tuple[int, int], g: tuple[int, int]):
    """A* on a 4-connected grid. `walls` is a [H, W] bool array.

    Returns the path as a list of (r, c) from s to g (inclusive), or None
    if unreachable.
    """
    h, w = walls.shape

    def heur(p):
        return abs(p[0] - g[0]) + abs(p[1] - g[1])

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (heur(s), counter, s))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    gscore: dict[tuple[int, int], int] = {s: 0}

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == g:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        cr, cc = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if walls[nr, nc]:
                continue
            tentative = gscore[cur] + 1
            n = (nr, nc)
            if tentative < gscore.get(n, 1 << 30):
                gscore[n] = tentative
                came_from[n] = cur
                counter += 1
                heapq.heappush(open_heap, (tentative + heur(n), counter, n))
    return None


def generate_maze(
    h: int, w: int, rng: np.random.Generator,
    wall_frac_lo: float = 0.30, wall_frac_hi: float = 0.50,
    min_path_len: int | None = None,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One valid maze as (x, y), each [h*w, 5], or None if every attempt failed.

    `min_path_len` is the minimum number of edges in the optimal S→G plan.
    Defaults to `max(h, w)` (Searchformer / Lehnert et al. 2024). HRM
    (arXiv:2506.21734) increases this to 110 for 30×30 mazes to make
    Maze-Hard meaningfully harder.
    """
    n_cells = h * w
    if min_path_len is None:
        min_path_len = max(h, w)

    for _ in range(max_attempts):
        wall_frac = float(rng.uniform(wall_frac_lo, wall_frac_hi))
        n_walls = int(round(wall_frac * n_cells))
        wall_indices = rng.choice(n_cells, size=n_walls, replace=False)
        walls = np.zeros(n_cells, dtype=bool)
        walls[wall_indices] = True
        walls = walls.reshape(h, w)

        non_wall = np.argwhere(~walls)
        if len(non_wall) < 2:
            continue
        # Random S and G from non-wall cells, distinct.
        idx = rng.choice(len(non_wall), size=2, replace=False)
        s = (int(non_wall[idx[0], 0]), int(non_wall[idx[0], 1]))
        g = (int(non_wall[idx[1], 0]), int(non_wall[idx[1], 1]))

        path = _astar(walls, s, g)
        if path is None or len(path) - 1 < min_path_len:
            continue

        # Build x and y in 5-channel encoding.
        x = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
        y = np.zeros((h, w, N_CHANNELS), dtype=np.float32)
        # Walls (in both x and y).
        x[walls, CH_WALL] = 1.0
        y[walls, CH_WALL] = 1.0
        # Free cells: free+path alive in x, free in y.
        free_mask = ~walls
        x[free_mask, CH_FREE] = 1.0
        x[free_mask, CH_PATH] = 1.0
        y[free_mask, CH_FREE] = 1.0
        # Path cells in y: replace free with path.
        for (r, c) in path[1:-1]:  # exclude S and G
            y[r, c] = 0.0
            y[r, c, CH_PATH] = 1.0
        # S and G singletons (override free+path on x and free on y).
        sx, sy = s
        gx, gy = g
        x[sx, sy] = 0.0
        x[sx, sy, CH_START] = 1.0
        x[gx, gy] = 0.0
        x[gx, gy, CH_GOAL] = 1.0
        y[sx, sy] = 0.0
        y[sx, sy, CH_START] = 1.0
        y[gx, gy] = 0.0
        y[gx, gy, CH_GOAL] = 1.0
        return x.reshape(n_cells, N_CHANNELS), y.reshape(n_cells, N_CHANNELS)

    return None




def hard_min_path_len(h: int, w: int) -> int:
    """Maze-Hard's minimum path length: 12.2% of the cells (110 at 30x30), at least max(h, w)."""
    return max(max(h, w), round(0.122 * h * w))


def _augment_d4_with_swap(
    x: np.ndarray, y: np.ndarray, h: int, w: int,
    rng: np.random.Generator, dihedral: bool, swap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Same D4 + S/G swap policy as maze_hard, generalized to (h, w)."""
    if swap and bool(rng.integers(0, 2)):
        x = x.copy()
        y = y.copy()
        x[:, [CH_START, CH_GOAL]] = x[:, [CH_GOAL, CH_START]]
        y[:, [CH_START, CH_GOAL]] = y[:, [CH_GOAL, CH_START]]
    if not dihedral:
        return x, y
    x = x.reshape(h, w, N_CHANNELS)
    y = y.reshape(h, w, N_CHANNELS)
    if h == w:
        k = int(rng.integers(0, 4))
        if k > 0:
            x = np.rot90(x, k=k, axes=(0, 1)).copy()
            y = np.rot90(y, k=k, axes=(0, 1)).copy()
    if bool(rng.integers(0, 2)):
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()
    return x.reshape(h * w, N_CHANNELS), y.reshape(h * w, N_CHANNELS)


@dataclass
class SyntheticMazeConfig:
    grid_h: int = 10
    grid_w: int = 10
    seed: int = 42
    wall_frac_lo: float = 0.30
    wall_frac_hi: float = 0.50

    # Difficulty mode. When `hard=True` (default), `min_path_len` defaults
    # to HRM-style: max( max(h,w), round(0.122 × h × w) ). The 12.2%-of-
    # total-cells ratio matches `sapientinc/maze-30x30-hard-1k`'s 110/900
    # cutoff and scales quadratically in n. When `hard=False`, the floor is
    # Searchformer's max(h, w). An explicit `min_path_len`
    # always overrides both.
    hard: bool = True
    min_path_len: int | None = None

    augment_dihedral: bool = True
    augment_swap_endpoints: bool = True

    prefetch_batches: int = 2
    batch_size: int = 64

    # Optional: how many distinct underlying mazes to draw from. None = unlimited
    # (every sample is freshly generated). When set, the dataset pre-generates
    # this many puzzles up front and cycles through them (useful for an "easy"
    # diagnostic regime where the model can memorize the underlying set).
    n_puzzles: int | None = None


class SyntheticMazeDataset:
    """Streaming synthetic maze dataset.

    `next_batch()` returns (x, y, is_sat) on CPU:
        x: [B, h*w, 5] float32 — powerset/lattice initial state
        y: [B, h*w, 5] float32 — singleton ground truth
        is_sat: [B] bool — always True
    """

    def __init__(self, cfg: SyntheticMazeConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._cells = cfg.grid_h * cfg.grid_w
        # Resolve effective min_path_len once: explicit > hard mode > default.
        if cfg.min_path_len is not None:
            self._min_path_len = cfg.min_path_len
        elif cfg.hard:
            self._min_path_len = hard_min_path_len(cfg.grid_h, cfg.grid_w)
        else:
            self._min_path_len = max(cfg.grid_h, cfg.grid_w)
        print(f"SyntheticMaze({cfg.grid_h}×{cfg.grid_w}, hard={cfg.hard}): "
              f"min_path_len={self._min_path_len}", flush=True)

        # Pre-generate finite pool if requested.
        self._pool: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cfg.n_puzzles is not None:
            self._pool = []
            print(f"Generating {cfg.n_puzzles} synthetic mazes "
                  f"({cfg.grid_h}×{cfg.grid_w}) …", flush=True)
            attempts = 0
            while len(self._pool) < cfg.n_puzzles:
                attempts += 1
                m = generate_maze(cfg.grid_h, cfg.grid_w, self.rng,
                                   cfg.wall_frac_lo, cfg.wall_frac_hi,
                                   min_path_len=self._min_path_len)
                if m is not None:
                    self._pool.append(m)
                if attempts > cfg.n_puzzles * 50:
                    raise RuntimeError(
                        f"could not generate {cfg.n_puzzles} valid mazes in "
                        f"{attempts} attempts — relax constraints?")
            self._order = self.rng.permutation(cfg.n_puzzles)
            self._pos = 0
            print(f"  generated {cfg.n_puzzles} (took {attempts} attempts)", flush=True)

        self._queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
            maxsize=cfg.prefetch_batches
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _next_sample(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._pool is not None:
            if self._pos >= len(self._pool):
                self._order = self.rng.permutation(len(self._pool))
                self._pos = 0
            x, y = self._pool[self._order[self._pos]]
            self._pos += 1
            x = x.copy()
            y = y.copy()
        else:
            m = generate_maze(self.cfg.grid_h, self.cfg.grid_w, self.rng,
                               self.cfg.wall_frac_lo, self.cfg.wall_frac_hi,
                               min_path_len=self._min_path_len)
            if m is None:
                # extreme bad luck — try again
                return self._next_sample()
            x, y = m

        x, y = _augment_d4_with_swap(
            x, y, self.cfg.grid_h, self.cfg.grid_w, self.rng,
            self.cfg.augment_dihedral, self.cfg.augment_swap_endpoints,
        )
        return x, y, True

    def _prefetch_loop(self):
        while not self._stop.is_set():
            bx, by, bs = [], [], []
            for _ in range(self.cfg.batch_size):
                x, y, is_sat = self._next_sample()
                bx.append(x)
                by.append(y)
                bs.append(is_sat)
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
