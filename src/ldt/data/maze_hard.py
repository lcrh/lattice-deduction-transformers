"""Maze-30x30-hard: download, encoding and cache.

Downloads sapientinc/maze-30x30-hard-1k from HuggingFace and caches it as a
.pt file of one-hot tensors (experiments/maze/data.py serves the training
samples).

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
- In the answer encoding, S and G stay labeled as start/goal (path cells are
  'o'), so they are singletons in `x` rather than carrying an extra `path`
  candidate. The solver deduces free-vs-path only on the blank cells.
"""

from __future__ import annotations

import csv
from pathlib import Path

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
