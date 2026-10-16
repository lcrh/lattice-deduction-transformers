import numpy as np

from baselines.hrm_trm import maze_scoring as ms

W, E, S, G, P = ms.WALL, ms.EMPTY, ms.START, ms.GOAL, ms.PATH

# 3x3 open maze, start top-left, goal bottom-right: shortest paths have 4 steps.
MAZE = np.array([[S, E, E],
                 [E, E, E],
                 [E, E, G]])


def test_any_shortest_path_is_lenient_correct():
    along_top = np.array([[S, P, P], [E, E, P], [E, E, G]])
    down_left = np.array([[S, E, E], [P, E, E], [P, P, G]])
    assert ms.lenient_ok(MAZE, along_top) and ms.lenient_ok(MAZE, down_left)


def test_longer_or_broken_paths_are_not():
    detour = np.array([[S, P, P], [E, P, P], [E, P, G]])        # not a simple path
    broken = np.array([[S, P, E], [E, E, P], [E, E, G]])
    assert not ms.lenient_ok(MAZE, detour) and not ms.lenient_ok(MAZE, broken)


def test_score_mazes_skips_padding_rows():
    pred = np.array([[S, P, P], [E, E, P], [E, E, G]])
    inputs = np.stack([MAZE.ravel(), np.zeros(9, dtype=int)])
    labels = np.stack([np.array([[S, E, E], [P, E, E], [P, P, G]]).ravel(), np.zeros(9, dtype=int)])
    preds = np.stack([pred.ravel(), np.zeros(9, dtype=int)])
    assert ms.score_mazes(inputs, labels, preds) == {"n": 1, "exact": 0, "lenient": 1}
