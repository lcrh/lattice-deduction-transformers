"""Maze-Hard failures in the style of the Maze-Hard example figure: one wrong
answer each at K = 1 and K = 512, drawn cell by cell against the closest
shortest path (maze_failure_mazes.pdf), and the excess path length of every
wrong answer of the checkpoints Table 3 reports (maze_failure_excess.pdf;
manifest `table3_maze`).

    uv run python -m paper.figures.maze_failure_panel [--copy-to DIR]
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import matplotlib.patches                      # noqa: E402
import numpy as np                              # noqa: E402

from paper.figures.maze_failures import analyse, closest_shortest_path, maze_test  # noqa: E402
from paper.tables.common import cli, load_rows, pdf_bytes, read_results, write_output  # noqa: E402
from paper.tables.table3_maze import _best_checkpoints  # noqa: E402

W = 30
SHOWN = (("K=1", 0, 540), ("K=512", 0, 706))     # (setting, seed, maze)
COLORS = {"wall": "#222222", "free": "#f3f3f3", "both": "#9fb7c9", "answer": "#e8743b",
          "shortest": "#4a7fb0", "start": "#2ca02c", "goal": "#d62728"}
SETTINGS = {"K=1": ("k1", "#5b5b5b"), "K=512": ("k512", "#b5b5b5")}
KEY = (("both", "answer and shortest path"), ("answer", "answer only"),
       ("shortest", "shortest path only"), ("start", "start"), ("goal", "goal"))


def rgb(name: str) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(COLORS[name]))


def draw_maze(ax, q: str, info: dict, title: str) -> None:
    true = set(closest_shortest_path(q, info["pred"]))
    pred = info["pred"]
    img = np.empty((W * W, 3))
    for k, ch in enumerate(q):
        name = ("wall" if ch == "#" else "start" if ch == "S" else "goal" if ch == "G"
                else "both" if k in pred and k in true else "answer" if k in pred
                else "shortest" if k in true else "free")
        img[k] = rgb(name)
    ax.imshow(img.reshape(W, W, 3), interpolation="nearest")
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])


def excess_counts(records: list[dict], questions: list[str]) -> dict[str, Counter]:
    """{setting: Counter of excess length (steps over the shortest path)} over wrong answers."""
    out = {}
    for label, (entry, _) in SETTINGS.items():
        c = Counter()
        for rec in _best_checkpoints(records, entry):
            for row in load_rows(rec):
                if row["status"] == "wrong":
                    info = analyse(questions[row["idx"]], row["solution"])
                    c["no path" if info["excess"] is None else info["excess"]] += 1
        out[label] = c
    return out


def draw_excess(ax, counts: dict[str, Counter]) -> None:
    keys = sorted({k for c in counts.values() for k in c if k != "no path"})
    keys += ["no path"] if any("no path" in c for c in counts.values()) else []
    x = np.arange(len(keys))
    width = 0.38
    for i, (label, c) in enumerate(counts.items()):
        ax.bar(x + (i - 0.5) * width, [c.get(k, 0) for k in keys], width,
               color=SETTINGS[label][1], label=f"{label} ({sum(c.values())} wrong)")
    ax.set_xticks(x, ["extra\ncells" if k == 0 else "no\npath" if k == "no path" else f"+{k}"
                      for k in keys], fontsize=7)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Steps over the shortest path", fontsize=8.5)
    ax.set_ylabel("Wrong answers (3 seeds)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1)


def main() -> None:
    args = cli("Maze-Hard failure panel.", default_manifest="table3_maze")
    records = read_results(args.manifest, args.results)
    questions = maze_test()
    fig = plt.figure(figsize=(4.4, 2.45))
    axes = [fig.add_axes(rect) for rect in ((0.0, 0.2, 0.49, 0.72), (0.51, 0.2, 0.49, 0.72))]
    for ax, (label, seed, idx) in zip(axes, SHOWN):
        entry = SETTINGS[label][0]
        rec = next(r for r in _best_checkpoints(records, entry) if r["seed"] == seed)
        row = next(r for r in load_rows(rec) if r["idx"] == idx)
        info = analyse(questions[idx], row["solution"])
        draw_maze(ax, questions[idx], info, f"{label}: {info['excess']} steps too long")
    fig.legend(handles=[matplotlib.patches.Patch(color=COLORS[c], label=t) for c, t in KEY],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=8, handlelength=1, columnspacing=1, borderaxespad=0)
    print(f"wrote {write_output('maze_failure_mazes.pdf', pdf_bytes(fig), args.out, args.copy_to)}")

    counts = excess_counts(records, questions)
    print({k: dict(v) for k, v in counts.items()})
    fig, ax = plt.subplots(figsize=(2.9, 2.45))
    draw_excess(ax, counts)
    fig.tight_layout(pad=0.3)
    print(f"wrote {write_output('maze_failure_excess.pdf', pdf_bytes(fig), args.out, args.copy_to)}")

if __name__ == "__main__":
    main()
