"""Snowflake Sudoku adapter for the shared evaluator (lattice_diffusion.eval).

Split `s` is the generated parquet `snowflake_<s>.parquet` in the data
directory; indices are row positions in it. Evaluation is zero-hint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lattice_diffusion.eval.tasks import EvalTask

from experiments.eval_common import build_solve_config, load_eval_model, result_rows
from experiments.snowflake.data import _load_puzzles, load_states
from experiments.snowflake.train import _build_state, _given_mask
from experiments.sudoku.solve import solve


def solution_string(state: torch.Tensor, mask: torch.Tensor) -> str:
    """Digits 1..6 of the in-puzzle cells in covering-grid order, '0' where
    the cell is not a singleton."""
    vocab = state[..., :6]
    single = vocab.sum(dim=-1) == 1
    digits = ((vocab.argmax(dim=-1) + 1) * single.long())[mask]
    return "".join(str(d) for d in digits.tolist())


class SnowflakeTask(EvalTask):
    """Snowflake Sudoku from the generated parquet splits."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)
        self._splits: dict[str, list[dict]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _puzzles(self, split: str) -> list[dict]:
        if split not in self._splits:
            self._splits[split] = _load_puzzles(str(Path(self.data_dir) / f"snowflake_{split}.parquet"))
        return self._splits[split]

    def split_size(self, split: str) -> int:
        import pyarrow.parquet as pq
        return pq.ParquetFile(Path(self.data_dir) / f"snowflake_{split}.parquet").metadata.num_rows

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool):
        return load_eval_model(checkpoint, dropout_p, self.device, compile)

    def warmup(self, model) -> None:
        from experiments.snowflake.config import SnowflakeExperiment
        self.solve_block(model, "test", np.arange(8), SnowflakeExperiment().solve_dict())

    def solve_block(self, model, split: str, indices: np.ndarray, solve_dict: dict) -> list[dict]:
        x, y, mask = load_states(self._puzzles(split), indices)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        cfg = build_solve_config(solve_dict)
        with torch.no_grad():
            res = solve(model, _build_state(x, mask), y, _given_mask(x), cfg,
                        in_puzzle_mask=mask, verbose=False)
        solution, mask = res.solution.cpu(), mask.cpu()
        # No rule check: the topology's constraints are not loaded here.
        solutions = [solution_string(solution[j], mask[j]) for j in range(len(indices))]
        return result_rows(res, indices, cfg, solutions, [None] * len(indices))
