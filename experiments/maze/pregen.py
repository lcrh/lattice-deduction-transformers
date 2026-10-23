"""Pregenerate a maze pool and its K sampled paths on Modal CPU workers, as parquet shards.

Training runs then read the cache through `MazeDataset`.

Usage:
    uv run modal run experiments/maze/pregen.py \
        --dataset synthetic --grid-size 30 --n-puzzles 10000 \
        --k-solutions 256 --seed 42 --workers 100

Each worker generates a deterministic chunk of puzzles, runs the K-paths
sampler, and writes its shard directly to a parquet file on the data volume.
The driver only writes the manifest — never holds full data in memory.

Layout under `/data/maze_pregen/`:
  {cache_key}.manifest.json          — list of shard paths + counts + meta
  {cache_key}_shards/shard_0000.parquet
  {cache_key}_shards/shard_0001.parquet
  ...

Parquet schema (one row per puzzle):
  x:         binary   (raw float32 bytes, shape [S, C])
  y:         binary   (raw float32 bytes, shape [S, C])
  solutions: binary   (raw float32 bytes, shape [K, S, C])
"""

import json
from pathlib import Path

import modal
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.maze.data import (
    PREGEN_CACHE_DIR, PREGEN_VERSION,
    MazeConfig, _build_k_solutions_with_metrics,
    _build_synthetic_canonical_pool, _load_maze_hard_canonical_pool,
    _pregen_cache_key, _pregen_cache_spec,
)
from ldt.modal.image import (
    DATA_MOUNT, data_volume, hf_secret, image,
)


app = modal.App("maze-pregen")


@app.function(
    image=image, cpu=4.0, memory=16384,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume},
    timeout=3600 * 4,  # 30x30 hard rejection rate is ~300:1; allow ample time
)
def pregen_chunk_shard(spec: dict, chunk_idx: int, chunk_size: int, shard_path: str) -> dict:
    """Generate one shard of puzzles and K paths, write it to `shard_path`, return its meta."""
    cfg = MazeConfig(
        dataset=spec["dataset"],
        cache_dir=DATA_MOUNT,
        split=spec.get("split", "train"),
        seed=spec["base_seed"] * 10000 + chunk_idx,
        n_puzzles=chunk_size,
        grid_size=spec.get("grid_size"),
        wall_frac_lo=spec.get("wall_frac_lo", 0.30),
        wall_frac_hi=spec.get("wall_frac_hi", 0.50),
        hard=spec.get("hard", True),
        k_solutions=spec["K"],
    )
    H = spec["H"]
    W = spec["W"]
    if cfg.dataset == "synthetic":
        x_np, y_np = _build_synthetic_canonical_pool(cfg, H, W)
    elif cfg.dataset == "maze_hard":
        x_np, y_np = _load_maze_hard_canonical_pool(cfg)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset!r}")

    K = spec["K"]
    sampler_seed = cfg.seed + 13
    sols_np = _build_k_solutions_with_metrics(
        x_np, y_np, K, H, W, sampler_seed,
    )

    n = x_np.shape[0]
    # Each puzzle's tensors → raw bytes. Faster than column-list types and
    # keeps decode trivial (np.frombuffer + reshape).
    x_bytes = [x_np[i].tobytes() for i in range(n)]
    y_bytes = [y_np[i].tobytes() for i in range(n)]
    sols_bytes = [sols_np[i].tobytes() for i in range(n)]
    table = pa.table({
        "x": pa.array(x_bytes, type=pa.binary()),
        "y": pa.array(y_bytes, type=pa.binary()),
        "solutions": pa.array(sols_bytes, type=pa.binary()),
    })
    Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, shard_path, compression="zstd")
    print(f"  [shard {chunk_idx}] wrote {n} puzzles → {shard_path}", flush=True)
    return {"path": shard_path, "n": n, "chunk_idx": chunk_idx}


@app.function(
    image=image, cpu=2.0, memory=8192,  # driver only writes the manifest — no concat
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume},
    timeout=3600 * 6,
)
def driver(
    spec: dict,
    n_puzzles: int,
    workers: int,
    cache_key: str,
    cfg_meta: dict,
):
    """Fan out shard generation, write manifest. Never aggregates data."""
    chunk_size = (n_puzzles + workers - 1) // workers
    n_chunks = (n_puzzles + chunk_size - 1) // chunk_size
    shards_dir = f"{DATA_MOUNT}/{PREGEN_CACHE_DIR}/{cache_key}_shards"
    manifest_path = f"{DATA_MOUNT}/{PREGEN_CACHE_DIR}/{cache_key}.manifest.json"
    Path(shards_dir).mkdir(parents=True, exist_ok=True)
    print(f"[driver] fanning out {n_chunks} workers × {chunk_size} puzzles → "
          f"{shards_dir}", flush=True)

    args = [
        (spec, i, chunk_size, f"{shards_dir}/shard_{i:04d}.parquet")
        for i in range(n_chunks)
    ]
    parts = list(pregen_chunk_shard.starmap(args))
    print(f"[driver] all {n_chunks} shards complete", flush=True)

    manifest = {
        "version": PREGEN_VERSION,
        "format": "parquet_shards",
        "spec": cfg_meta,
        "shards": [
            {"path": p["path"], "n": p["n"], "idx": p["chunk_idx"]}
            for p in sorted(parts, key=lambda x: x["chunk_idx"])
        ],
        "meta": {
            "K": spec["K"],
            "H": spec["H"],
            "W": spec["W"],
            "n_puzzles": sum(p["n"] for p in parts),
            "n_shards": n_chunks,
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    data_volume.commit()
    print(f"[driver] manifest at {manifest_path} "
          f"(total {manifest['meta']['n_puzzles']} puzzles across "
          f"{n_chunks} shards)", flush=True)
    return {"manifest_path": manifest_path, "n_puzzles": manifest["meta"]["n_puzzles"]}


@app.local_entrypoint()
def entrypoint(
    dataset: str = "synthetic",
    grid_size: int | None = None,
    n_puzzles: int = 10000,
    k_solutions: int = 1,
    seed: int = 1,
    wall_frac_lo: float = 0.30,
    wall_frac_hi: float = 0.50,
    hard: bool = True,
    cache_suffix: str = "",
    workers: int = 100,
    split: str = "train",
):
    if dataset == "synthetic" and grid_size is None:
        raise ValueError("synthetic pregen requires --grid-size")

    if dataset == "synthetic":
        H = W = grid_size
    elif dataset == "maze_hard":
        from experiments.maze.data import MAZE_HARD_H, MAZE_HARD_W
        H, W = MAZE_HARD_H, MAZE_HARD_W
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    cfg = MazeConfig(
        dataset=dataset,
        cache_dir=DATA_MOUNT,
        split=split,
        seed=seed,
        n_puzzles=n_puzzles,
        grid_size=grid_size,
        wall_frac_lo=wall_frac_lo,
        wall_frac_hi=wall_frac_hi,
        hard=hard,
        k_solutions=k_solutions,
        cache_suffix=cache_suffix,
    )
    cache_key = _pregen_cache_key(cfg)
    spec_dict = _pregen_cache_spec(cfg)

    print(f"Pregen plan (parquet shards):")
    print(f"  dataset={dataset}  grid={H}x{W}  n_puzzles={n_puzzles}  K={k_solutions}")
    print(f"  seed={seed}  workers={workers}  hard={hard}")
    print(f"  cache_key={cache_key}")
    print(f"  cache_key spec: {spec_dict}")

    spec = {
        "dataset": dataset,
        "split": split,
        "base_seed": seed,
        "K": k_solutions,
        "grid_size": grid_size,
        "wall_frac_lo": wall_frac_lo,
        "wall_frac_hi": wall_frac_hi,
        "hard": hard,
        "H": H,
        "W": W,
    }

    if dataset == "maze_hard":
        workers = 1
        print(f"  (maze_hard: forcing workers=1; HF set has fixed N)")

    result = driver.remote(spec, n_puzzles, workers, cache_key, spec_dict)
    print(f"\n[entrypoint] driver returned: {result}", flush=True)
