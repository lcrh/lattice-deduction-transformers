"""Sudoku-Extreme adapter for the shared evaluator (lattice_diffusion.eval).

Every Sudoku-Extreme puzzle is satisfiable and evaluation is zero-hint with
no data augmentation, so an eval sample is exactly the cached (x, y) pair;
we index the cache directly instead of going through the training sampler.
"""

from __future__ import annotations

import numpy as np
import torch

from lattice_diffusion.data.sudoku_extreme import _download_and_cache
from lattice_diffusion.eval.tasks import EvalTask

from experiments.eval_common import build_solve_config, load_eval_model, result_rows
from experiments.sudoku.solve import solve


def grid_string(solution: torch.Tensor) -> str:
    """[81, 9] lattice -> 81-char digit string, '0' for non-singleton cells."""
    single = solution.sum(dim=-1) == 1
    digits = (solution.argmax(dim=-1) + 1) * single.long()
    return "".join(str(d) for d in digits.tolist())


def is_valid_solution(grid: str, puzzle: torch.Tensor) -> bool:
    """Rule check: complete grid, givens kept, rows/cols/boxes all-different."""
    if len(grid) != 81 or "0" in grid:
        return False
    cells = [int(ch) for ch in grid]
    given = puzzle.sum(dim=-1) == 1
    given_digits = (puzzle.argmax(dim=-1) + 1).tolist()
    for i in range(81):
        if given[i] and cells[i] != given_digits[i]:
            return False
    full = set(range(1, 10))
    for r in range(9):
        if set(cells[r * 9:(r + 1) * 9]) != full:
            return False
    for c in range(9):
        if set(cells[c::9]) != full:
            return False
    for br in range(3):
        for bc in range(3):
            box = [cells[(br * 3 + i) * 9 + bc * 3 + j] for i in range(3) for j in range(3)]
            if set(box) != full:
                return False
    return True


class SudokuTask(EvalTask):
    """Sudoku-Extreme, read straight from the cached dataset tensors."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)
        self._pools: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pool(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Memory-mapped: the train split (for sudoku_val) is 2.8 GB.
        if split not in self._pools:
            data = torch.load(_download_and_cache(self.data_dir, split),
                              map_location="cpu", weights_only=True, mmap=True)
            self._pools[split] = (data["x"], data["y"])
        return self._pools[split]

    def split_size(self, split: str) -> int:
        return self._pool(split)[0].shape[0]

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool):
        return load_eval_model(checkpoint, dropout_p, self.device, compile)

    def warmup(self, model) -> None:
        from experiments.sudoku.config import SudokuExperiment
        self.solve_block(model, "test", np.arange(8), SudokuExperiment().solve_dict())

    def solve_block(self, model, split: str, indices: np.ndarray, solve_dict: dict) -> list[dict]:
        x_all, y_all = self._pool(split)
        idx = torch.as_tensor(indices, dtype=torch.long)
        x = x_all.index_select(0, idx).to(self.device).float()
        y = y_all.index_select(0, idx).to(self.device).float()
        given_mask = x.sum(dim=-1) == 1
        cfg = build_solve_config(solve_dict)
        with torch.no_grad():
            res = solve(model, x, y, given_mask, cfg, verbose=False)

        solution, x = res.solution.cpu(), x.cpu()
        grids = [grid_string(solution[j]) for j in range(len(indices))]
        verified = [is_valid_solution(grids[j], x[j]) for j in range(len(indices))]
        return result_rows(res, indices, cfg, grids, verified)
