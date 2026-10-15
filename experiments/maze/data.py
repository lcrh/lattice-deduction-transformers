"""Unified dataset wrapper for maze.

Picks between `MazeHardDataset` (HuggingFace 30×30 maze-hard) and
`SyntheticMazeDataset` (on-the-fly generated mazes at any grid size)
based on `MazeConfig.dataset`. Exposes:

  - `H, W` — spatial grid dims (30×30 for maze_hard; configurable for
    synthetic; default 10×10).
  - `N_CHANNELS` = 5 (wall, free, start, goal, path) — fixed by the
    encoding shared between both loaders.
  - `next_batch()` returning `(x, y, is_sat)` matching sudoku's
    expected 3-tuple contract.

run.py uses (H, W, N_CHANNELS) to build the LoopedTransformerConfig
spatial dims; everything else flows through the standard pipeline.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Literal

import torch

import numpy as np

from lattice_diffusion.data.maze_hard import (
    CH_FREE, CH_GOAL, CH_PATH, CH_START, CH_WALL,
    GRID_H as MAZE_HARD_H,
    GRID_W as MAZE_HARD_W,
    N_CHANNELS,
    MazeHardConfig,
    MazeHardDataset,
    _download_and_cache as _maze_hard_download_and_cache,
)
from lattice_diffusion.data.maze_synthetic import (
    SyntheticMazeConfig,
    SyntheticMazeDataset,
    _augment_d4_with_swap,
    generate_maze,
    hard_min_path_len,
)


PREGEN_CACHE_DIR = "maze_pregen"
PREGEN_VERSION = 1


@dataclass
class MazeConfig:
    """Unified config selecting between maze_hard and synthetic.

    Use `dataset='maze_hard'` for the HF 30×30 set (grid_h/grid_w are
    ignored, fixed at 30×30). Use `dataset='synthetic'` for on-the-fly
    generation at the requested grid_h/grid_w (default 10×10).
    """
    dataset: Literal["maze_hard", "synthetic"] = "maze_hard"

    # Generic.
    seed: int = 42
    batch_size: int = 64

    # maze_hard-specific.
    cache_dir: str = "data"
    split: str = "train"
    augment_swap_endpoints: bool = True

    # Square grid side length. `None` defers to the dataset's natural size
    # (30 for maze_hard, 10 for synthetic). Setting an explicit `grid_size`
    # alongside `dataset='maze_hard'` errors — that dataset is fixed at
    # 30x30 by the HF source. For now we only support square synthetic
    # grids — `grid_size` is used for both h and w.
    grid_size: int | None = None
    wall_frac_lo: float = 0.30
    wall_frac_hi: float = 0.50
    hard: bool = True

    # Pool size — meaning differs by dataset:
    #   maze_hard: subset of HF puzzles (None = all).
    #   synthetic: cycle through a fixed pool of N pre-generated puzzles
    #     (None = unlimited fresh stream).
    n_puzzles: int | None = None

    # Common to both loaders.
    augment_dihedral: bool = True
    prefetch_batches: int = 2

    # diagnostic mode (maze_hard only) — replace the puzzle with an
    # S-to-G straight line for quick sanity checks.
    simplify_to_straight_line: bool = False

    # Pregen cache invalidation knob. The cache key hashes
    # (config, seed, cache_suffix), so bumping this string forces a fresh
    # build even with identical config — useful for ablations or after
    # the sampler/generator code changes.
    cache_suffix: str = ""

    # Number of ground-truth solutions per puzzle (K). K=1 (default) uses
    # the canonical A* GT. K>1 enables the α(surviving K paths)
    # multi-alive supervision: per puzzle, sample K-1 additional uniform
    # shortest paths from the all-shortest-paths DAG. The trainer
    # recomputes α(surviving) each step to drive BCE + gt_conflict against
    # the dynamic lattice.
    k_solutions: int = 1


def grid_dims(cfg: MazeConfig) -> tuple[int, int]:
    """Return (H, W) for the model config — mode-aware.

    `maze_hard` is fixed at 30x30 by the HF dataset; passing a different
    `grid_size` explicitly while `dataset='maze_hard'` is an error
    (silently ignoring it would be confusing).
    """
    if cfg.dataset == "maze_hard":
        if cfg.grid_size is not None and cfg.grid_size != MAZE_HARD_H:
            raise ValueError(
                f"dataset='maze_hard' is fixed at {MAZE_HARD_H}x{MAZE_HARD_W} "
                f"(got grid_size={cfg.grid_size}). Use dataset='synthetic' "
                f"for configurable grid sizes."
            )
        return MAZE_HARD_H, MAZE_HARD_W
    # synthetic — default to 10 if unset.
    side = cfg.grid_size if cfg.grid_size is not None else 10
    return side, side


# -----------------------------------------------------------------------------
# Pregen cache (canonical pool + K-solutions) on the modal volume.
#
# Why: K-paths sampling over a finite pool is deterministic given (puzzles,
# K, sampler_seed). Caching it once on the data volume avoids re-running the
# Python sampler at every training launch (~1-3s for 200 puzzles × K=64;
# multiplies up for larger pools and K). Maze pool generation is also slow
# at large grid sizes (rejection-resample to hit min_path_len). Both are
# pre-computed in canonical frame; the dataset's prefetch thread applies
# fresh per-sample augmentation.

def _pregen_cache_spec(cfg: MazeConfig) -> dict:
    """Canonical config snapshot used for cache-key hashing. Only fields that
    change the underlying canonical pool/solutions are included; augmentation
    knobs (which are applied per-sample post-cache) are excluded.
    """
    spec: dict = {
        "version": PREGEN_VERSION,
        "dataset": cfg.dataset,
        "n_puzzles": cfg.n_puzzles,
        "seed": cfg.seed,
        "k_solutions": int(cfg.k_solutions),
        "cache_suffix": cfg.cache_suffix,
    }
    if cfg.dataset == "synthetic":
        side = cfg.grid_size if cfg.grid_size is not None else 10
        spec.update({
            "grid_size": side,
            "wall_frac_lo": cfg.wall_frac_lo,
            "wall_frac_hi": cfg.wall_frac_hi,
            "hard": cfg.hard,
        })
    elif cfg.dataset == "maze_hard":
        spec.update({
            "split": cfg.split,
            "simplify_to_straight_line": cfg.simplify_to_straight_line,
        })
    return spec


def _pregen_cache_key(cfg: MazeConfig) -> str:
    spec = _pregen_cache_spec(cfg)
    s = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _build_synthetic_canonical_pool(
    cfg: MazeConfig, H: int, W: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate `cfg.n_puzzles` canonical synthetic mazes (no augmentation).
    Returns (x_np, y_np) of shapes [N, H*W, C]. Logs mazes/sec progress.
    """
    n = cfg.n_puzzles
    if n is None:
        raise ValueError("synthetic pregen requires n_puzzles to be set")
    rng = np.random.default_rng(cfg.seed)
    if cfg.hard:
        min_pl = hard_min_path_len(H, W)
    else:
        min_pl = max(H, W)
    x_np = np.zeros((n, H * W, N_CHANNELS), dtype=np.float32)
    y_np = np.zeros((n, H * W, N_CHANNELS), dtype=np.float32)
    print(f"  [pregen pool] generating {n} synthetic {H}×{W} mazes "
          f"(hard={cfg.hard}, min_path_len={min_pl}) …", flush=True)
    t0 = time.time()
    n_attempts = 0
    i = 0
    log_every = max(1, n // 20)
    while i < n:
        n_attempts += 1
        m = generate_maze(H, W, rng, cfg.wall_frac_lo, cfg.wall_frac_hi,
                          min_path_len=min_pl)
        if m is None:
            # 30x30 hard has ~300:1 rejection — cap needs to be generous so
            # workers don't bail out on slow chunks. 500x is conservative.
            if n_attempts > n * 500:
                raise RuntimeError(
                    f"could not generate {n} valid {H}×{W} mazes in "
                    f"{n_attempts} attempts"
                )
            continue
        x, y = m
        x_np[i] = x.reshape(H * W, N_CHANNELS)
        y_np[i] = y.reshape(H * W, N_CHANNELS)
        i += 1
        if i % log_every == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"    {i}/{n} mazes  ({rate:.1f}/s, "
                  f"{n_attempts} attempts)", flush=True)
    return x_np, y_np


def _load_maze_hard_canonical_pool(
    cfg: MazeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load canonical maze_hard puzzles from the HF cache. Applies
    `n_puzzles` subset selection (seed-driven) to match `MazeHardDataset`.
    """
    cache_path = _maze_hard_download_and_cache(cfg.cache_dir, cfg.split)
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    x_full = data["x"].numpy().astype(np.float32)  # [N_total, 900, 5]
    y_full = data["y"].numpy().astype(np.float32)
    n_total = x_full.shape[0]
    rng = np.random.default_rng(cfg.seed)
    if cfg.n_puzzles is not None and cfg.n_puzzles < n_total:
        idx = rng.choice(n_total, size=cfg.n_puzzles, replace=False)
        idx.sort()
        x_full = x_full[idx]
        y_full = y_full[idx]
    print(f"  [pregen pool] loaded {x_full.shape[0]} maze_hard puzzles "
          f"({cfg.split} split, n_total={n_total})", flush=True)
    return x_full, y_full


def _build_k_solutions_with_metrics(
    pool_x: np.ndarray, pool_y: np.ndarray,
    K: int, H: int, W: int, sampler_seed: int,
) -> np.ndarray:
    """Sample K solutions per puzzle. Returns [N, K, H*W, C] float32.
    Logs progress (puzzles/sec) every 5% of the pool.
    """
    n = pool_x.shape[0]
    rng = random.Random(sampler_seed)
    out = np.zeros((n, K, H * W, N_CHANNELS), dtype=np.float32)
    if K <= 1:
        out[:, 0] = pool_y.reshape(n, H * W, N_CHANNELS)
        return out
    print(f"  [pregen K-paths] sampling K={K} solutions for {n} puzzles …",
          flush=True)
    t0 = time.time()
    log_every = max(1, n // 20)
    for i in range(n):
        x_grid = pool_x[i].reshape(H, W, N_CHANNELS)
        y_grid = pool_y[i].reshape(H, W, N_CHANNELS)
        out[i] = sample_k_solutions(x_grid, y_grid, K, rng)
        if (i + 1) % log_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    {i+1}/{n} puzzles  (K={K}; {rate:.2f} puzzles/s)",
                  flush=True)
    return out


def _load_pregen_manifest(cfg: MazeConfig) -> dict | None:
    """If a parquet-shard manifest exists for this cache key, return its
    parsed dict. Otherwise return None.
    """
    cache_key = _pregen_cache_key(cfg)
    manifest_path = (Path(cfg.cache_dir) / PREGEN_CACHE_DIR
                     / f"{cache_key}.manifest.json")
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest


def _build_or_load_pregen(
    cfg: MazeConfig, H: int, W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns canonical (pool_x, pool_y, pool_solutions) — `pool_solutions`
    has shape [N, K, H*W, C]. Loads from `{cache_dir}/maze_pregen/{key}.pt`
    if present and valid; otherwise generates and saves.

    Used for the single-file pregen format. For parquet-shard pregen
    output, callers detect via `_load_pregen_manifest` first and use the
    sharded streaming path in `MazeDataset._init_sharded_pool` instead,
    which never loads the full pool into RAM.
    """
    K = max(1, int(cfg.k_solutions))
    cache_key = _pregen_cache_key(cfg)
    cache_path = Path(cfg.cache_dir) / PREGEN_CACHE_DIR / f"{cache_key}.pt"

    if cache_path.exists():
        print(f"  [pregen] cache hit (mmap): {cache_path}", flush=True)
        # mmap=True keeps the cache as memory-mapped tensors so we never load
        # the full 50+ GB into RAM. Pages are faulted in on access.
        data = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        pool_x = data["x"]
        pool_y = data["y"]
        pool_solutions = data["solutions"]
        n = pool_x.shape[0]
        # Sanity: cache might be stale if K was bumped post-build. Validate
        # K dim before handing off.
        if pool_solutions.shape[1] != K:
            print(f"  [pregen] cache K dim mismatch ({pool_solutions.shape[1]} "
                  f"!= {K}); rebuilding", flush=True)
        elif pool_x.shape[0] != pool_y.shape[0] or pool_y.shape[0] != n:
            print(f"  [pregen] cache row-count mismatch; rebuilding", flush=True)
        else:
            print(f"    n={n}  K={K}  shape={tuple(pool_solutions.shape)}",
                  flush=True)
            return pool_x, pool_y, pool_solutions

    print(f"  [pregen] cache miss; building (will save to {cache_path})",
          flush=True)
    spec = _pregen_cache_spec(cfg)
    print(f"    spec: {spec}", flush=True)

    if cfg.dataset == "synthetic":
        pool_x_np, pool_y_np = _build_synthetic_canonical_pool(cfg, H, W)
    elif cfg.dataset == "maze_hard":
        pool_x_np, pool_y_np = _load_maze_hard_canonical_pool(cfg)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset!r}")

    # K-solutions sampler RNG seed — kept separate from `cfg.seed` so the
    # sampler stream doesn't entangle with the data-loader and model
    # init seeds.
    sampler_seed = cfg.seed + 13
    pool_solutions_np = _build_k_solutions_with_metrics(
        pool_x_np, pool_y_np, K, H, W, sampler_seed,
    )

    pool_x = torch.from_numpy(pool_x_np)
    pool_y = torch.from_numpy(pool_y_np)
    pool_solutions = torch.from_numpy(pool_solutions_np)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "x": pool_x,
        "y": pool_y,
        "solutions": pool_solutions,
        "meta": {
            "version": PREGEN_VERSION,
            "spec": spec,
            "K": K,
            "H": H,
            "W": W,
            "n_puzzles": pool_x.shape[0],
        },
    }, cache_path)
    print(f"  [pregen] cached to {cache_path} "
          f"(size: x={tuple(pool_x.shape)}, sols={tuple(pool_solutions.shape)})",
          flush=True)
    return pool_x, pool_y, pool_solutions


class MazeDataset:
    """Dispatching wrapper. Yields `(x, solutions, is_sat)` per
    `next_batch()` call where `solutions: [B, K, S, 5]` carries K candidate
    ground-truth paths per puzzle.

    Two modes:
      - Finite-pool (n_puzzles is set, OR dataset='maze_hard'): builds or
        loads a canonical pool + K-solutions cache via `_build_or_load_pregen`.
        A worker thread samples random indices, applies fresh per-sample
        augmentation, and emits batches. K-paths sampling cost is paid once
        per (config, K, seed) tuple, not per run.
      - Streaming (dataset='synthetic' with n_puzzles=None): falls back to
        the inner SyntheticMazeDataset + on-the-fly K-paths sampler in a
        prefetch thread. Pregen doesn't apply (the pool is unbounded).
    """

    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        self.K = max(1, int(cfg.k_solutions))
        self._H, self._W = grid_dims(cfg)

        if cfg.dataset == "synthetic" and cfg.simplify_to_straight_line:
            raise ValueError(
                "simplify_to_straight_line is a maze_hard-only diagnostic mode; "
                "set dataset='maze_hard' or unset the flag."
            )

        # Decide finite-pool vs streaming. maze_hard is always finite (HF
        # set is the entire pool). Synthetic is finite when n_puzzles is set.
        self._finite_pool = (cfg.dataset == "maze_hard"
                             or (cfg.dataset == "synthetic" and cfg.n_puzzles is not None))

        if self._finite_pool:
            # Prefer the parquet-shard streaming path if a manifest exists.
            self._manifest = _load_pregen_manifest(cfg)
            if self._manifest is not None:
                self._init_sharded_pool()
            else:
                self._init_finite_pool()
        else:
            self._init_streaming()
            self._manifest = None

    # ---- sharded-pool path (parquet streaming) -----------------------------
    def _init_sharded_pool(self):
        """Stream from parquet shards: one shard at a time on a background
        thread, shuffled within shard, no full-pool RAM hit.
        """
        import pyarrow.parquet as pq  # local import: pyarrow only loaded on the sharded path
        self._pq = pq

        manifest = self._manifest
        cfg = self.cfg
        # Sanity check K matches.
        if int(manifest["meta"]["K"]) != self.K:
            print(f"  [pregen] manifest K={manifest['meta']['K']} != cfg K={self.K}; "
                  f"falling back to in-process build", flush=True)
            self._manifest = None
            self._init_finite_pool()
            return
        self._shard_paths = [s["path"] for s in manifest["shards"]]
        self._shard_n = [int(s["n"]) for s in manifest["shards"]]
        self._N = sum(self._shard_n)
        print(f"  [pregen] shard manifest hit: {len(self._shard_paths)} shards, "
              f"{self._N} total puzzles, K={self.K}", flush=True)

        self._aug_rng = np.random.default_rng(cfg.seed + 7)
        self._queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
            maxsize=cfg.prefetch_batches
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sharded_pool_loop, daemon=True)
        self._thread.start()
        self._inner = None  # no inner dataset in sharded mode

    def _load_shard(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read one parquet shard back into numpy arrays."""
        H, W, K = self._H, self._W, self.K
        S = H * W
        C = N_CHANNELS
        table = self._pq.read_table(path)
        n = len(table)
        x_col = table["x"].to_pylist()
        y_col = table["y"].to_pylist()
        s_col = table["solutions"].to_pylist()
        x_np = np.empty((n, S, C), dtype=np.float32)
        y_np = np.empty((n, S, C), dtype=np.float32)
        sols_np = np.empty((n, K, S, C), dtype=np.float32)
        for i in range(n):
            x_np[i] = np.frombuffer(x_col[i], dtype=np.float32).reshape(S, C)
            y_np[i] = np.frombuffer(y_col[i], dtype=np.float32).reshape(S, C)
            sols_np[i] = np.frombuffer(s_col[i], dtype=np.float32).reshape(K, S, C)
        return x_np, y_np, sols_np

    def _sharded_pool_loop(self):
        """Background loop: load shards into a multi-shard buffer (so batch_size
        > shard_size still works), shuffle, emit batches with augmentation.
        """
        cfg = self.cfg
        H, W, K = self._H, self._W, self.K
        bs = cfg.batch_size
        n_shards = len(self._shard_paths)
        shard_order = self._aug_rng.permutation(n_shards)
        next_shard_in_order = 0
        # Multi-shard buffer: keep loading until we have at least max(bs, 2*bs)
        # rows, then shuffle and emit batches. Leftover rolls into next refill.
        buf_x: list[np.ndarray] = []
        buf_y: list[np.ndarray] = []
        buf_sols: list[np.ndarray] = []
        buf_n = 0

        while not self._stop.is_set():
            # Refill buffer until we have enough for a batch.
            while buf_n < bs and not self._stop.is_set():
                if next_shard_in_order >= n_shards:
                    shard_order = self._aug_rng.permutation(n_shards)
                    next_shard_in_order = 0
                shard_idx = int(shard_order[next_shard_in_order])
                next_shard_in_order += 1
                try:
                    sx, sy, ss = self._load_shard(self._shard_paths[shard_idx])
                except Exception as e:
                    print(f"  [shard load error] {self._shard_paths[shard_idx]}: {e}", flush=True)
                    continue
                buf_x.append(sx); buf_y.append(sy); buf_sols.append(ss)
                buf_n += sx.shape[0]
            if self._stop.is_set():
                return

            # Concat buffer + shuffle.
            x_all = np.concatenate(buf_x, axis=0)
            y_all = np.concatenate(buf_y, axis=0)
            sols_all = np.concatenate(buf_sols, axis=0)
            order = self._aug_rng.permutation(buf_n)
            cursor = 0
            while cursor + bs <= buf_n and not self._stop.is_set():
                idx = order[cursor:cursor + bs]
                cursor += bs
                x_b = x_all[idx].copy()
                y_b = y_all[idx].copy()
                sols_b = sols_all[idx].copy()
                # Per-puzzle aug (same as _finite_pool_loop).
                for i in range(bs):
                    if cfg.augment_swap_endpoints and bool(self._aug_rng.integers(0, 2)):
                        x_b[i, :, [CH_START, CH_GOAL]] = x_b[i, :, [CH_GOAL, CH_START]]
                        y_b[i, :, [CH_START, CH_GOAL]] = y_b[i, :, [CH_GOAL, CH_START]]
                        sols_b[i, :, :, [CH_START, CH_GOAL]] = sols_b[i, :, :, [CH_GOAL, CH_START]]
                    if cfg.augment_dihedral:
                        rot_k = int(self._aug_rng.integers(0, 4)) if H == W else 0
                        flip = bool(self._aug_rng.integers(0, 2))
                        if rot_k or flip:
                            x_grid = x_b[i].reshape(H, W, N_CHANNELS)
                            y_grid = y_b[i].reshape(H, W, N_CHANNELS)
                            sols_grid = sols_b[i].reshape(K, H, W, N_CHANNELS)
                            if rot_k:
                                x_grid = np.rot90(x_grid, k=rot_k, axes=(0, 1))
                                y_grid = np.rot90(y_grid, k=rot_k, axes=(0, 1))
                                sols_grid = np.rot90(sols_grid, k=rot_k, axes=(1, 2))
                            if flip:
                                x_grid = np.flip(x_grid, axis=1)
                                y_grid = np.flip(y_grid, axis=1)
                                sols_grid = np.flip(sols_grid, axis=2)
                            x_b[i] = x_grid.reshape(H * W, N_CHANNELS)
                            y_b[i] = y_grid.reshape(H * W, N_CHANNELS)
                            sols_b[i] = sols_grid.reshape(K, H * W, N_CHANNELS)
                tx = torch.from_numpy(x_b)
                tsols = torch.from_numpy(sols_b)
                tsat = torch.ones(bs, dtype=torch.bool)
                try:
                    self._queue.put((tx, tsols, tsat), timeout=1.0)
                except Exception:
                    if self._stop.is_set():
                        return
            # Roll any leftover (cursor..buf_n) into next refill.
            leftover_idx = order[cursor:]
            if len(leftover_idx) > 0:
                buf_x = [x_all[leftover_idx]]
                buf_y = [y_all[leftover_idx]]
                buf_sols = [sols_all[leftover_idx]]
                buf_n = len(leftover_idx)
            else:
                buf_x = []; buf_y = []; buf_sols = []; buf_n = 0
            del x_all, y_all, sols_all  # free big concat tensor

    # ---- finite-pool path (with pregen cache) -------------------------------
    def _init_finite_pool(self):
        cfg = self.cfg
        # Load or build canonical pool (with K-solutions baked in).
        self._pool_x, self._pool_y, self._pool_solutions = _build_or_load_pregen(
            cfg, self._H, self._W,
        )
        # Numpy views for the aug worker (avoid repeated .numpy() copies).
        self._pool_x_np = self._pool_x.numpy()
        self._pool_y_np = self._pool_y.numpy()
        self._pool_sols_np = self._pool_solutions.numpy()
        self._N = self._pool_x.shape[0]
        self._aug_rng = np.random.default_rng(cfg.seed + 7)

        self._queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
            maxsize=cfg.prefetch_batches
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._finite_pool_loop, daemon=True)
        self._thread.start()
        # No inner dataset in this mode.
        self._inner = None

    def _finite_pool_loop(self):
        cfg = self.cfg
        H, W, K = self._H, self._W, self.K
        bs = cfg.batch_size
        # Sequential cycling with modulo — disk-friendly when the pool is
        # mmap-backed (avoids random-access page thrashing on multi-GB caches).
        # Modulo handles the n_puzzles < batch_size case (small probe pools).
        # Per-batch augmentation provides diversity.
        cursor = 0
        while not self._stop.is_set():
            idx = (np.arange(cursor, cursor + bs)) % self._N
            cursor = (cursor + bs) % self._N
            x_b = self._pool_x_np[idx].copy()              # [B, S, C]
            y_b = self._pool_y_np[idx].copy()              # [B, S, C]
            sols_b = self._pool_sols_np[idx].copy()        # [B, K, S, C]
            for i in range(bs):
                # Sample one shared aug per puzzle (so x, y, all K solutions
                # remain mutually consistent).
                if cfg.augment_swap_endpoints and bool(self._aug_rng.integers(0, 2)):
                    # S/G channel swap (per-cell, no spatial transform).
                    x_b[i, :, [CH_START, CH_GOAL]] = x_b[i, :, [CH_GOAL, CH_START]]
                    y_b[i, :, [CH_START, CH_GOAL]] = y_b[i, :, [CH_GOAL, CH_START]]
                    sols_b[i, :, :, [CH_START, CH_GOAL]] = sols_b[i, :, :, [CH_GOAL, CH_START]]
                if cfg.augment_dihedral:
                    rot_k = int(self._aug_rng.integers(0, 4)) if H == W else 0
                    flip = bool(self._aug_rng.integers(0, 2))
                    if rot_k or flip:
                        x_grid = x_b[i].reshape(H, W, N_CHANNELS)
                        y_grid = y_b[i].reshape(H, W, N_CHANNELS)
                        sols_grid = sols_b[i].reshape(K, H, W, N_CHANNELS)
                        if rot_k:
                            x_grid = np.rot90(x_grid, k=rot_k, axes=(0, 1))
                            y_grid = np.rot90(y_grid, k=rot_k, axes=(0, 1))
                            sols_grid = np.rot90(sols_grid, k=rot_k, axes=(1, 2))
                        if flip:
                            x_grid = np.flip(x_grid, axis=1)
                            y_grid = np.flip(y_grid, axis=1)
                            sols_grid = np.flip(sols_grid, axis=2)
                        x_b[i] = x_grid.reshape(H * W, N_CHANNELS)
                        y_b[i] = y_grid.reshape(H * W, N_CHANNELS)
                        sols_b[i] = sols_grid.reshape(K, H * W, N_CHANNELS)
            tx = torch.from_numpy(x_b)
            tsols = torch.from_numpy(sols_b)
            tsat = torch.ones(bs, dtype=torch.bool)
            try:
                self._queue.put((tx, tsols, tsat), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    # ---- streaming path (synthetic with n_puzzles=None) ---------------------
    def _init_streaming(self):
        cfg = self.cfg
        side = cfg.grid_size if cfg.grid_size is not None else 10
        self._inner = SyntheticMazeDataset(SyntheticMazeConfig(
            grid_h=side,
            grid_w=side,
            seed=cfg.seed,
            wall_frac_lo=cfg.wall_frac_lo,
            wall_frac_hi=cfg.wall_frac_hi,
            hard=cfg.hard,
            augment_dihedral=cfg.augment_dihedral,
            augment_swap_endpoints=cfg.augment_swap_endpoints,
            batch_size=cfg.batch_size,
            n_puzzles=cfg.n_puzzles,
            prefetch_batches=cfg.prefetch_batches,
        ))
        # K>1: prefetch thread that pulls (x, y, is_sat) from inner and emits
        # (x, solutions, is_sat). K=1: inline pass-through (no extra thread).
        self._k_thread = None
        if self.K > 1:
            self._k_queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
                maxsize=cfg.prefetch_batches
            )
            self._k_stop = threading.Event()
            self._k_rng = random.Random(cfg.seed + 13)
            self._k_thread = threading.Thread(target=self._k_prefetch_loop, daemon=True)
            self._k_thread.start()

    def _k_prefetch_loop(self):
        while not self._k_stop.is_set():
            x, y, is_sat = self._inner.next_batch()
            x_np = x.numpy()
            y_np = y.numpy()
            sols_np = sample_k_solutions_batch(
                x_np, y_np, self.K, self._H, self._W, self._k_rng,
            )
            sols = torch.from_numpy(sols_np)
            try:
                self._k_queue.put((x, sols, is_sat), timeout=1.0)
            except Exception:
                if self._k_stop.is_set():
                    break

    # ---- public API ---------------------------------------------------------
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(x, solutions, is_sat)` where `solutions: [B, K, S, 5]`."""
        if self._finite_pool:
            return self._queue.get()
        if self.K > 1:
            return self._k_queue.get()
        x, y, is_sat = self._inner.next_batch()
        return x, y.unsqueeze(1), is_sat

    def close(self):
        if self._finite_pool:
            self._stop.set()
        else:
            if getattr(self, "_k_thread", None) is not None:
                self._k_stop.set()
            self._inner.close()


# -----------------------------------------------------------------------------
# Maze-aware correctness rescoring.
#
# `solve()` flags `correct` via cell-by-cell argmax equality against the GT
# (sudoku-style strict match). For maze, ANY valid minimal path counts —
# the model could pick a different optimal route through the maze and still
# be right. This rescorer relaxes the check to:
#   1. Predicted path-cell count == GT path-cell count (GT is minimal by
#      construction in maze-hard / synthetic-hard).
#   2. The traversable subgraph (path | S | G cells, 4-connected) is
#      connected — every path cell is reachable from S, and G is reached.
#   3. There's exactly one S and one G in the prediction (sanity).
#
# Called post-hoc on `SolveResult.solution`. Timeouts (solver never self-
# accepted) stay `correct=False` regardless. Solved-but-incorrect become
# `wrong=True`.


def _bfs_connected_with_g(grid: np.ndarray, sr: int, sc: int, gr: int, gc: int) -> tuple[bool, int]:
    """Return (g_reachable, n_cells_visited). Traverses cells where grid is True
    (i.e. the path/S/G mask). 4-connected."""
    H, W = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    visited[sr, sc] = True
    stack = [(sr, sc)]
    n_visited = 1
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                n_visited += 1
                stack.append((nr, nc))
    return bool(visited[gr, gc]), int(n_visited)


def maze_classify(
    solutions: torch.Tensor,   # [P, S, C] solver final state (one-hot when solved)
    targets: torch.Tensor,     # [P, S, C] GT one-hot
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-puzzle classification of the model's predicted path.

    Returns (`valid`, `minimal`, `gt_exact_match`) — all `[P]` bool, defined
    on the union of all puzzles (timeouts will fail all three checks because
    their state isn't a singleton solution).
    - `valid[p] = True` iff the predicted path forms a single connected
      component (with S and G as endpoints) reachable from S to G, with no
      detached path islands and exactly one S and one G.
    - `minimal[p] = True` iff `valid[p]` AND the predicted path-cell count
      equals the GT path-cell count (the GT is minimal by construction in
      maze-hard / synthetic-hard, so length-match implies optimality).
    - `gt_exact_match[p] = True` iff every cell where GT is path is also
      path in pred AND every cell where GT is free is also free in pred.
      Walls/S/G are pinned by `given_mask` so they're identical between
      pred and GT for any solved puzzle, which means a path/free agreement
      across the whole grid is the full cell-by-cell match.

    The five buckets a caller will care about (see `rescore`):
      CORRECT_GT      = solved & valid & minimal & gt_exact_match
      CORRECT_ALT     = solved & valid & minimal & ~gt_exact_match
      WRONG_VALID     = solved & valid & ~minimal   (legal route, but longer than optimal)
      WRONG_INVALID   = solved & ~valid             (broken / disconnected path)
      TIMEOUT         = ~solved
    Plus an umbrella `WRONG = WRONG_VALID | WRONG_INVALID = solved & ~correct`.
    """
    P, S, C = solutions.shape
    assert S == H * W, f"S={S} != H*W={H*W}"
    assert C == N_CHANNELS, f"C={C} != N_CHANNELS={N_CHANNELS}"
    pred = solutions.argmax(dim=-1).cpu().numpy().reshape(P, H, W)        # int 0..4
    gt = targets.argmax(dim=-1).cpu().numpy().reshape(P, H, W)
    valid = np.zeros(P, dtype=bool)
    minimal = np.zeros(P, dtype=bool)
    gt_exact_match = np.zeros(P, dtype=bool)
    expected_path_count = (gt == CH_PATH).sum(axis=(1, 2))                 # [P]
    pred_path_count = (pred == CH_PATH).sum(axis=(1, 2))                   # [P]
    # gt_exact_match: cell-by-cell equality of "is path?" mask between pred
    # and GT over the whole grid. Walls/S/G are pinned by given_mask so this
    # is equivalent to a full argmax-equality check on solved puzzles.
    pred_is_path = (pred == CH_PATH)
    gt_is_path = (gt == CH_PATH)
    cell_match_all = (pred_is_path == gt_is_path).all(axis=(1, 2))         # [P]
    for p in range(P):
        s_pos = np.argwhere(pred[p] == CH_START)
        g_pos = np.argwhere(pred[p] == CH_GOAL)
        if s_pos.shape[0] != 1 or g_pos.shape[0] != 1:
            continue
        sr, sc = int(s_pos[0, 0]), int(s_pos[0, 1])
        gr, gc = int(g_pos[0, 0]), int(g_pos[0, 1])
        traversable = np.isin(pred[p], (CH_START, CH_GOAL, CH_PATH))
        n_traversable = int(traversable.sum())
        g_reach, n_visited = _bfs_connected_with_g(traversable, sr, sc, gr, gc)
        # Valid: reachable AND no detached path islands.
        if g_reach and n_visited == n_traversable:
            valid[p] = True
            if pred_path_count[p] == expected_path_count[p]:
                minimal[p] = True
                if cell_match_all[p]:
                    gt_exact_match[p] = True
    valid_t = torch.from_numpy(valid).to(solutions.device)
    minimal_t = torch.from_numpy(minimal).to(solutions.device)
    gt_exact_t = torch.from_numpy(gt_exact_match).to(solutions.device)
    return valid_t, minimal_t, gt_exact_t


def rescore(
    res,                       # SolveResult from experiments.sudoku.solve.solve
    targets: torch.Tensor,     # [P, S, C] GT one-hot
    H: int,
    W: int,
):
    """Relax `res.correct` / `res.wrong` to maze-aware semantics in-place
    on the provided SolveResult. Returns
    `(n_correct, n_correct_gt, n_correct_alt, n_wrong, n_wrong_valid, n_wrong_invalid, n_timeout)`:
      - `n_correct = n_correct_gt + n_correct_alt` — umbrella "any correct"
      - `n_correct_gt`     = solved & valid & minimal & gt_exact_match
      - `n_correct_alt`    = solved & valid & minimal & ~gt_exact_match
        (alternate optimal-length path with different cells than GT)
      - `n_wrong = n_wrong_valid + n_wrong_invalid` — umbrella "solved but not correct"
      - `n_wrong_valid`    = solved & valid & ~minimal (legal route but longer than
        optimal — not counted as correct)
      - `n_wrong_invalid`  = solved & ~valid (broken / disconnected / multi-S/G)
      - `n_timeout`        = ~solved (model never self-accepted)

    The 5-way partition CORRECT_GT / CORRECT_ALT / WRONG_VALID / WRONG_INVALID /
    TIMEOUT covers every puzzle exactly once.

    On the SolveResult: `res.correct` is set to umbrella correctness
    (`solved & valid & minimal`), `res.wrong` is set to umbrella wrong
    (`solved & ~correct`).
    """
    valid, minimal, gt_exact = maze_classify(res.solution, targets, H, W)
    solved = res.solved
    correct = solved & valid & minimal
    correct_gt = correct & gt_exact
    correct_alt = correct & ~gt_exact
    wrong_valid = solved & valid & ~minimal
    wrong_invalid = solved & ~valid
    new_wrong = solved & ~correct  # = wrong_valid | wrong_invalid
    res.correct = correct
    res.wrong = new_wrong
    n_correct = int(correct.sum().item())
    n_correct_gt = int(correct_gt.sum().item())
    n_correct_alt = int(correct_alt.sum().item())
    n_wrong_valid = int(wrong_valid.sum().item())
    n_wrong_invalid = int(wrong_invalid.sum().item())
    n_wrong = int(new_wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    # Stash classify results so callers can format per-puzzle labels.
    res._maze_valid = valid
    res._maze_minimal = minimal
    res._maze_gt_exact = gt_exact
    return (n_correct, n_correct_gt, n_correct_alt,
            n_wrong, n_wrong_valid, n_wrong_invalid, n_timeout)


# -----------------------------------------------------------------------------
# K-path sampler (K-solutions α-target)
#
# For each puzzle: enumerate the all-shortest-paths DAG (cells where d_S + d_G
# == D), count paths to G via reverse-topological scan, then sample K paths
# uniformly. Each path is materialized as a one-hot solution tensor [S, C]
# matching the dataset's encoding (walls/S/G as singletons, free cells as
# {free} or {path} per the path bitmap).
#
# Cost per puzzle: 2 BFS (O(N)) + 1 path-count pass (O(N)) + K weighted
# samples (O(K · D)). At 30×30, K=64: ~5ms total in Python.


def _bfs_distances(walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """4-connected BFS. Returns int distance from `start` to each non-wall
    cell; np.iinfo(np.int32).max for unreachable / wall cells."""
    H, W = walls.shape
    INF = np.iinfo(np.int32).max
    dist = np.full((H, W), INF, dtype=np.int32)
    dist[start] = 0
    queue: list[tuple[int, int]] = [start]
    head = 0
    while head < len(queue):
        r, c = queue[head]; head += 1
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc] and dist[nr, nc] == INF:
                dist[nr, nc] = d + 1
                queue.append((nr, nc))
    return dist


def _count_paths_to_g(d_g: np.ndarray, on_dag: np.ndarray, g_pos: tuple[int, int]) -> dict:
    """Number of shortest paths from each on-DAG cell to G.

    `paths_to_g[c] = sum over neighbors n with d_g[n] = d_g[c] - 1 of
    paths_to_g[n]`. Base case `paths_to_g[g] = 1`. Uses Python ints
    (arbitrary precision) so big mazes don't overflow int64.
    """
    H, W = d_g.shape
    paths: dict[tuple[int, int], int] = {}
    paths[g_pos] = 1
    # Iterate cells by increasing d_g so children (d_g-1) computed first.
    by_dg: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            if on_dag[r, c]:
                by_dg.append((int(d_g[r, c]), r, c))
    by_dg.sort()
    for _, r, c in by_dg:
        if (r, c) == g_pos:
            continue
        total = 0
        my_dg = int(d_g[r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and on_dag[nr, nc] and int(d_g[nr, nc]) == my_dg - 1:
                total += paths.get((nr, nc), 0)
        paths[(r, c)] = total
    return paths


def _bigint_uniform(rng: random.Random, total: int) -> int:
    """Return a uniform random Python int in [0, total). Wraps
    `random.Random.randrange`, which internally does getrandbits + rejection
    over arbitrary-precision Python ints (unlike `np.random.Generator.integers`,
    capped at int64). Faster than a hand-rolled rejection loop because the
    inner getrandbits + comparison runs in C.
    """
    return rng.randrange(total)


def _sample_one_path(
    d_g: np.ndarray,
    on_dag: np.ndarray,
    paths_to_g: dict,
    s_pos: tuple[int, int],
    g_pos: tuple[int, int],
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Walk from S to G; at each step pick next cell weighted by
    `paths_to_g[n] / sum`. Yields a uniform random shortest path.
    Returns the set of intermediate `path` cells (excluding S and G).

    Uses arbitrary-precision Python ints throughout so dense DAGs (where
    shortest-path counts can exceed int64) sample correctly.
    """
    H, W = d_g.shape
    cur = s_pos
    visited = []
    while cur != g_pos:
        my_dg = int(d_g[cur])
        # Candidates: neighbors on DAG with d_g one closer.
        cands: list[tuple[int, int]] = []
        weights: list[int] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < H and 0 <= nc < W and on_dag[nr, nc] and int(d_g[nr, nc]) == my_dg - 1:
                cands.append((nr, nc))
                weights.append(paths_to_g.get((nr, nc), 0))
        total = sum(weights)
        if total == 0:
            # Shouldn't happen if BFS/DAG were computed correctly.
            return set()
        # Sample weighted with bigint-safe arithmetic.
        r = _bigint_uniform(rng, total)
        acc = 0
        chosen = cands[-1]
        for cand, w in zip(cands, weights):
            acc += w
            if r < acc:
                chosen = cand
                break
        cur = chosen
        if cur != g_pos:
            visited.append(cur)
    return set(visited)


def sample_k_solutions(
    x_grid: np.ndarray,    # [H, W, C]
    y_grid: np.ndarray,    # [H, W, C] — the original A* GT
    K: int,
    rng: random.Random,
) -> np.ndarray:
    """Return K solutions [K, H*W, C] for one puzzle.

    `solutions[0]` is always the original A* GT (`y_grid` flattened). The
    remaining K-1 are uniform shortest-path samples from the all-paths DAG;
    duplicates can occur for K large vs total-path-count and are harmless
    (α(surviving) just collapses).

    For K=1, returns a single [1, H*W, C] tensor wrapping `y_grid`.
    """
    H, W, C = x_grid.shape
    out = np.zeros((K, H * W, C), dtype=np.float32)
    out[0] = y_grid.reshape(H * W, C)
    if K <= 1:
        return out

    walls = x_grid[..., CH_WALL] > 0.5
    s_positions = np.argwhere(x_grid[..., CH_START] > 0.5)
    g_positions = np.argwhere(x_grid[..., CH_GOAL] > 0.5)
    if s_positions.shape[0] != 1 or g_positions.shape[0] != 1:
        # Malformed puzzle — fall back to replicating y_grid.
        for k in range(1, K):
            out[k] = y_grid.reshape(H * W, C)
        return out
    s_pos = (int(s_positions[0, 0]), int(s_positions[0, 1]))
    g_pos = (int(g_positions[0, 0]), int(g_positions[0, 1]))

    d_s = _bfs_distances(walls, s_pos)
    d_g = _bfs_distances(walls, g_pos)
    INF = np.iinfo(np.int32).max
    if d_s[g_pos] == INF:
        for k in range(1, K):
            out[k] = y_grid.reshape(H * W, C)
        return out
    D = int(d_s[g_pos])
    on_dag = (d_s.astype(np.int64) + d_g.astype(np.int64) == D) & ~walls
    paths_to_g = _count_paths_to_g(d_g, on_dag, g_pos)

    # Build a base solution (walls / S / G singletons; free cells -> CH_FREE
    # singleton). Each sampled path then flips its path-cells from CH_FREE
    # to CH_PATH.
    base = np.zeros((H, W, C), dtype=np.float32)
    base[..., CH_WALL] = walls.astype(np.float32)
    s_sg_mask = (x_grid[..., CH_START] > 0.5)
    g_sg_mask = (x_grid[..., CH_GOAL] > 0.5)
    base[..., CH_START] = s_sg_mask.astype(np.float32)
    base[..., CH_GOAL] = g_sg_mask.astype(np.float32)
    free_mask = ~walls & ~s_sg_mask & ~g_sg_mask
    base[..., CH_FREE] = free_mask.astype(np.float32)

    for k in range(1, K):
        path_cells = _sample_one_path(d_g, on_dag, paths_to_g, s_pos, g_pos, rng)
        sol = base.copy()
        for r, c in path_cells:
            sol[r, c, CH_FREE] = 0.0
            sol[r, c, CH_PATH] = 1.0
        out[k] = sol.reshape(H * W, C)
    return out


def sample_k_solutions_batch(
    x_batch: np.ndarray,   # [B, H*W, C]
    y_batch: np.ndarray,   # [B, H*W, C]
    K: int,
    H: int,
    W: int,
    rng: random.Random,
) -> np.ndarray:
    """Vectorize `sample_k_solutions` over a batch. Returns `[B, K, H*W, C]`."""
    B, S, C = x_batch.shape
    out = np.zeros((B, K, S, C), dtype=np.float32)
    for i in range(B):
        x_i = x_batch[i].reshape(H, W, C)
        y_i = y_batch[i].reshape(H, W, C)
        out[i] = sample_k_solutions(x_i, y_i, K, rng)
    return out
