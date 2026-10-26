"""Maze-Hard failure analysis: the wrong answers of the checkpoints Table 3
reports (each seed's best checkpoint, manifest `table3_maze`), with a
true shortest path drawn over each.

For every wrong answer it prints whether the path connects S to G, its length
against the shortest path, and how many of its cells lie on some shortest path.
The figure shows one example of each kind of failure (`SHOWN`); the true
shortest path drawn is the one that shares the most cells with the model's path.

    uv run python -m paper.figures.maze_failures [--copy-to DIR]
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
from matplotlib.colors import ListedColormap    # noqa: E402

from paper.tables.common import cli, load_rows, pdf_bytes, read_results, write_output  # noqa: E402
from paper.tables.table3_maze import _best_checkpoints  # noqa: E402

N, W = 900, 30
WALL, FREE, START, GOAL, PATH = 0, 1, 2, 3, 4      # ldt.data.maze_hard channels
# (setting, seed, maze): two failures per setting, of different kinds.
SHOWN = [("K=1", 0, 540), ("K=1", 1, 52), ("K=512", 0, 299), ("K=512", 0, 706)]
START_COLOR, GOAL_COLOR = "#1a9850", "#7b3294"


def maze_test() -> list[str]:
    path = next(Path.home().glob(".cache/huggingface/hub/datasets--sapientinc--maze-30x30-hard-1k/"
                                 "snapshots/*/test.csv"))
    return [r["question"] for r in csv.DictReader(path.open())]


def neighbours(u: int):
    r, c = divmod(u, W)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= r + dr < W and 0 <= c + dc < W:
            yield (r + dr) * W + c + dc


def bfs(src: int, ok) -> dict[int, int]:
    dist, dq = {src: 0}, deque([src])
    while dq:
        u = dq.popleft()
        for v in neighbours(u):
            if v not in dist and ok(v):
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def closest_shortest_path(q: str, pred: set[int]) -> list[int]:
    """The shortest S-G path sharing the most cells with `pred` (DP over the shortest-path DAG)."""
    s, g = q.index("S"), q.index("G")
    ds, dg = bfs(s, lambda v: q[v] != "#"), bfs(g, lambda v: q[v] != "#")
    best = ds[g]
    on = {u for u in ds if u in dg and ds[u] + dg[u] == best}
    score, nxt = {g: 0}, {}
    for u in sorted(on, key=lambda u: -ds[u]):
        if u == g:
            continue
        succ = [v for v in neighbours(u) if v in on and ds[v] == ds[u] + 1]
        v = max(succ, key=lambda v: score[v])
        score[u], nxt[u] = score[v] + (u in pred), v
    path, u = [s], s
    while u != g:
        u = nxt[u]
        path.append(u)
    return path


def analyse(q: str, solution: str) -> dict:
    s, g = q.index("S"), q.index("G")
    pred = {k for k, ch in enumerate(solution) if int(ch) == PATH}
    ds, dg = bfs(s, lambda v: q[v] != "#"), bfs(g, lambda v: q[v] != "#")
    best = ds[g]
    reach = bfs(s, lambda v: v in pred or v == g).get(g)
    on_shortest = sum(1 for k in pred if k in ds and k in dg and ds[k] + dg[k] == best)
    return {"shortest": best, "path_cells": len(pred), "connected": reach is not None,
            "length": reach, "excess": None if reach is None else reach - best,
            "on_shortest": on_shortest, "pred": pred}


def draw(ax, q: str, info: dict, title: str) -> None:
    grid = np.array([WALL if ch == "#" else FREE for ch in q]).reshape(W, W)
    ax.imshow(grid, cmap=ListedColormap(["0.25", "white"]), interpolation="nearest")
    true = closest_shortest_path(q, info["pred"])
    tr, tc = np.divmod(np.array(true), W)
    ax.plot(tc, tr, "-", color="C0", lw=2.4, alpha=0.55, solid_capstyle="round")
    off = sorted(info["pred"] - set(true))       # model cells off the drawn shortest path
    pr, pc = np.divmod(np.array(off, dtype=int), W)
    ax.scatter(pc, pr, s=5, color="C3", marker="s", linewidths=0)
    for ch, mk, color in (("S", "o", START_COLOR), ("G", "*", GOAL_COLOR)):
        r, c = divmod(q.index(ch), W)
        ax.scatter([c], [r], s=45 if mk == "o" else 90, color=color, marker=mk, zorder=5, edgecolors="white",
                   linewidths=0.5)
    ax.set_title(title, fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = cli("Maze-Hard failure analysis.", default_manifest="table3_maze")
    records = read_results(args.manifest, args.results)
    questions = maze_test()
    failures = []
    for train_entry, label in (("k1", "K=1"), ("k512", "K=512")):
        for rec in _best_checkpoints(records, train_entry):
            for row in load_rows(rec):
                if row["status"] != "correct":
                    info = analyse(questions[row["idx"]], row["solution"]) if row["status"] == "wrong" else None
                    failures.append((label, rec["seed"], row["idx"], row["status"], info))
    wrong = [f for f in failures if f[3] == "wrong"]
    print(f"{len(failures)} failures ({len(wrong)} wrong, {len(failures) - len(wrong)} timeouts)")
    for label, seed, idx, _, info in wrong:
        print(f"  {label} seed {seed} maze {idx}: connected={info['connected']} "
              f"length {info['length']} vs shortest {info['shortest']} (excess {info['excess']}), "
              f"{info['on_shortest']}/{info['path_cells']} path cells on some shortest path")
    by_key = {(f[0], f[1], f[2]): f for f in wrong}
    shown = [by_key[k] for k in SHOWN]
    fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.9))
    for ax, (label, seed, idx, _, info) in zip(axes.flat, shown):
        if info["excess"] is None:
            what = "does not reach G"
        elif info["excess"] == 0:
            what = f"shortest + {info['path_cells'] - info['length'] + 1} extra cells"
        else:
            what = f"{info['excess']} steps too long"
        draw(ax, questions[idx], info, f"{label}, maze {idx}: {what}")
    fig.tight_layout(pad=0.3)
    print(f"wrote {write_output('maze_failures.pdf', pdf_bytes(fig), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
