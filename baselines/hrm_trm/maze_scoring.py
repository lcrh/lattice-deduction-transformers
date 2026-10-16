"""Maze scoring shared by the HRM and TRM evals: exact match and lenient (any shortest path).

Shipped into the eval images next to the scripts, so it only needs numpy.
"""

from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

# Token ids of HRM/TRM's maze datasets (CHARSET "# SGo" -> 1..5, PAD = 0).
WALL, EMPTY, START, GOAL, PATH = 1, 2, 3, 4, 5

_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def bfs_dist(passable: np.ndarray, src: tuple, dst: tuple) -> int | None:
    """Shortest 4-connected path length (in steps) over passable cells; None if unreachable."""
    h, w = passable.shape
    seen = {src}
    queue = deque([(src, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == dst:
            return d
        for dr, dc in _STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return None


def lenient_ok(inp: np.ndarray, pred: np.ndarray) -> bool:
    """True iff pred's path cells form a simple start-to-goal path of optimal length."""
    passable = inp != WALL
    starts, goals = np.argwhere(inp == START), np.argwhere(inp == GOAL)
    if len(starts) != 1 or len(goals) != 1:
        return False
    src, dst = tuple(starts[0]), tuple(goals[0])

    cells = set(map(tuple, np.argwhere((pred == START) | (pred == GOAL) | (pred == PATH))))
    if src not in cells or dst not in cells or any(not passable[c] for c in cells):
        return False

    def neighbours(cell):
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in _STEPS if (r + dr, c + dc) in cells]

    # A simple path: endpoints of degree 1, interior cells of degree 2, one component.
    degree = {c: len(neighbours(c)) for c in cells}
    if degree[src] != 1 or degree[dst] != 1:
        return False
    if any(degree[c] != 2 for c in cells if c not in (src, dst)):
        return False
    seen, queue = {src}, deque([src])
    while queue:
        for n in neighbours(queue.popleft()):
            if n not in seen:
                seen.add(n)
                queue.append(n)
    return len(seen) == len(cells) and len(cells) - 1 == bfs_dist(passable, src, dst)


def score_mazes(inputs: np.ndarray, labels: np.ndarray, pred_ids: np.ndarray) -> dict:
    """{n, exact, lenient} over the non-padding rows of flattened square mazes."""
    side = int(round(inputs.shape[1] ** 0.5))
    n = exact = lenient = 0
    for inp, lab, prd in zip(inputs, labels, pred_ids):
        inp, lab, prd = (a.reshape(side, side) for a in (inp, lab, prd))
        if not ((inp == START).any() and (inp == GOAL).any()):
            continue        # padding row
        n += 1
        exact += int(np.array_equal(prd, lab))
        lenient += int(lenient_ok(inp, prd))
    return {"n": n, "exact": exact, "lenient": lenient}


def print_scores(model: str, scores: dict) -> None:
    n = scores["n"]
    print(f"{model} maze-30x30-hard ({n} test puzzles): "
          f"exact {scores['exact']}/{n} = {100 * scores['exact'] / n:.2f}%, "
          f"lenient {scores['lenient']}/{n} = {100 * scores['lenient'] / n:.2f}%", flush=True)


def write_sweep(checkpoints: list[str], results: list[dict], out: str) -> None:
    """Save and print {step, n, exact, lenient} per checkpoint of a training run."""
    rows = sorted(({"step": int(os.path.basename(c).removeprefix("step_")), **r}
                   for c, r in zip(checkpoints, results)), key=lambda r: r["step"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[sweep] wrote {out}", flush=True)
    for r in rows:
        print(f"  step {r['step']:>7}: exact={100 * r['exact'] / r['n']:.2f}%  "
              f"lenient={100 * r['lenient'] / r['n']:.2f}%", flush=True)


def _cache_path(checkpoint: str) -> str:
    return os.path.join(os.path.dirname(checkpoint), "maze_scores",
                        os.path.basename(checkpoint) + ".json")


def cached_scores(checkpoint: str, settings: dict) -> dict | None:
    """Saved scores of `checkpoint` under identical eval `settings`, if any."""
    path = _cache_path(checkpoint)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        saved = json.load(f)
    return saved["scores"] if saved["settings"] == settings else None


def save_scores(checkpoint: str, settings: dict, scores: dict) -> None:
    """Save a checkpoint's scores next to it, so a rerun sweep skips it."""
    path = _cache_path(checkpoint)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"settings": settings, "scores": scores}, f)
