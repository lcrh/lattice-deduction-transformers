"""Maze adapter for the shared evaluator.

Split "test" is the Maze-Hard test split; split "synthetic" is a seed-200
pool of generated mazes described by the solve dict's `maze_data`.

A puzzle is correct when the predicted path is any shortest start-to-goal
path. Each row's `label` (CORRECT_GT / CORRECT_ALT / WRONG_VALID /
WRONG_INVALID / TIMEOUT) also allows exact-match scoring against the
reference path.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from lattice_diffusion.data.maze_hard import _download_and_cache
from lattice_diffusion.eval.tasks import EvalTask, TaskBlock

from experiments.eval_common import build_solve_config, eval_device, load_eval_model, result_rows
from experiments.maze.data import (
    MazeConfig, _build_or_load_pregen, grid_dims, rescore, shortest_path_check,
)
from experiments.sudoku.solve import solve


class MazeTask(EvalTask):
    """Maze-Hard and synthetic mazes, scored on any shortest path."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)
        self._pools: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.device = eval_device()

    def _data_cfg(self, split: str, maze_data: dict | None) -> MazeConfig:
        if split == "test":
            return MazeConfig(dataset="maze_hard", cache_dir=self.data_dir, split="test")
        if split != "synthetic" or maze_data is None:
            raise ValueError(f"unknown maze split {split!r}")
        return MazeConfig(
            dataset="synthetic", cache_dir=self.data_dir, seed=200,
            grid_size=maze_data["grid_size"], hard=maze_data["hard"],
            wall_frac_lo=maze_data["wall_frac_lo"], wall_frac_hi=maze_data["wall_frac_hi"],
            n_puzzles=maze_data["n_puzzles"],
        )

    def _pool(self, split: str, maze_data: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        key = split if split == "test" else json.dumps(maze_data, sort_keys=True)
        if key not in self._pools:
            cfg = self._data_cfg(split, maze_data)
            if split == "test":
                data = torch.load(_download_and_cache(self.data_dir, "test"),
                                  map_location="cpu", weights_only=True)
                self._pools[key] = (data["x"], data["y"])
            else:
                x, y, _ = _build_or_load_pregen(cfg, *grid_dims(cfg))
                self._pools[key] = (x, y)
        return self._pools[key]

    def split_size(self, split: str) -> int:
        return self._pool(split, None)[0].shape[0]

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool):
        return load_eval_model(checkpoint, dropout_p, self.device, compile)

    def default_solve(self) -> dict:
        from experiments.maze.config import MazeExperiment
        return MazeExperiment().solve_dict()

    def warmup(self, model) -> None:
        self.solve_block(model, "test", np.arange(4), self.default_solve())

    def solve_block(self, model, split: str, indices: np.ndarray, solve_dict: dict) -> TaskBlock:
        solve_dict = dict(solve_dict)
        maze_data = solve_dict.pop("maze_data")
        x_all, y_all = self._pool(split, maze_data)
        idx = torch.as_tensor(indices, dtype=torch.long)
        x = x_all.index_select(0, idx).to(self.device).float()
        y = y_all.index_select(0, idx).to(self.device).float()
        h, w = grid_dims(self._data_cfg(split, maze_data))
        cfg = build_solve_config(solve_dict)
        given_mask = x.sum(dim=-1) == 1
        with torch.no_grad():
            res = solve(model, x, y, given_mask, cfg, is_correct=shortest_path_check(h, w))
        outcome = rescore(res, y, h, w)

        valid, minimal, exact = (outcome.valid.tolist(), outcome.minimal.tolist(),
                                 outcome.gt_exact.tolist())
        cells = res.solution.argmax(dim=-1).cpu()
        solutions = ["".join(str(c) for c in cells[j].tolist()) for j in range(len(indices))]
        labels = []
        for j, timeout in enumerate(res.timeouts.tolist()):
            if timeout:
                labels.append("TIMEOUT")
            elif minimal[j]:
                labels.append("CORRECT_GT" if exact[j] else "CORRECT_ALT")
            else:
                labels.append("WRONG_VALID" if valid[j] else "WRONG_INVALID")
        return result_rows(res, indices, cfg, solutions, minimal, given_mask,
                           [{"label": label} for label in labels])
