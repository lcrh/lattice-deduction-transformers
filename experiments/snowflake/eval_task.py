"""Snowflake Sudoku adapter for the shared evaluator.

A split is a generated parquet file (data.split_file); indices are its rows.
Evaluation is zero-hint. Rows carry the puzzle's order, and an `orders` index
filter restricts a list to some orders.
"""

from __future__ import annotations

import numpy as np
import torch

from lattice_diffusion.eval.tasks import EvalTask, TaskBlock

from experiments.eval_common import (
    build_solve_config, eval_device, load_eval_model, result_rows, unwrap,
)
from experiments.snowflake.data import (
    BLANK, cell_index, load_puzzles, load_states, puzzle_orders, split_file, split_name,
)
from experiments.snowflake.train import N_CHANNELS, VOCAB, _build_state, _given_mask
from experiments.sudoku.solve import solve


def solution_string(state: torch.Tensor, mask: torch.Tensor) -> str:
    """Digits of the in-puzzle cells in covering-grid order, '0' where not a singleton."""
    vocab = state[..., :VOCAB]
    single = vocab.sum(dim=-1) == 1
    digits = ((vocab.argmax(dim=-1) + 1) * single.long())[mask]
    return "".join(str(d) for d in digits.tolist())


def is_valid_solution(state: torch.Tensor, puzzle_rec: dict) -> bool:
    """Every cell a singleton, givens kept, every constraint group all-different."""
    vocab = state[..., :VOCAB]
    topology = puzzle_rec["topology"]
    digit = {}
    for cell_id, cp in topology["cell_positions"].items():
        cell = vocab[cell_index(cp)]
        if int(cell.sum()) != 1:
            return False
        digit[int(cell_id)] = int(cell.argmax()) + 1
    for cell_id, given in enumerate(puzzle_rec["puzzle"]):
        if given != BLANK and digit[cell_id] != given:
            return False
    for group in topology["constraints"]:
        values = [digit[int(c)] for c in group["cells"]]
        if len(set(values)) != len(values):
            return False
    return True


def _uses_group_channels(model) -> bool:
    return unwrap(model).cfg.n_channels > N_CHANNELS


class SnowflakeTask(EvalTask):
    """Snowflake Sudoku from the generated parquet splits."""

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)
        self._pools: dict[str, list[dict]] = {}
        self.device = eval_device()

    def _puzzles(self, split: str) -> list[dict]:
        if split not in self._pools:
            self._pools[split] = load_puzzles(split_file(self.data_dir, split))
        return self._pools[split]

    def split_size(self, split: str) -> int:
        import pyarrow.parquet as pq
        return pq.ParquetFile(split_file(self.data_dir, split)).metadata.num_rows

    def filter_indices(self, split: str, indices: np.ndarray, index_filter: dict) -> np.ndarray:
        unknown = set(index_filter) - {"orders"}
        if unknown:
            raise ValueError(f"unsupported snowflake index_filter keys {sorted(unknown)}")
        orders = puzzle_orders(self._puzzles(split))
        return indices[np.isin(orders[indices], index_filter["orders"])]

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool):
        return load_eval_model(checkpoint, dropout_p, self.device, compile)

    def default_solve(self) -> dict:
        from experiments.snowflake.config import SnowflakeExperiment
        return SnowflakeExperiment().solve_dict()

    def warmup(self, model) -> None:
        from experiments.snowflake.config import SnowflakeExperiment
        self.solve_block(model, split_name(SnowflakeExperiment().data.dataset, "test"),
                         np.arange(8), self.default_solve())

    def solve_block(self, model, split: str, indices: np.ndarray, solve_dict: dict) -> TaskBlock:
        puzzles = self._puzzles(split)
        # Constraint-group IDs are random per load; draw them from the block's
        # seeded torch RNG so a block's rows are reproducible.
        rng = (np.random.default_rng(torch.initial_seed() % 2**63)
               if _uses_group_channels(model) else None)
        x, y, mask, groups = load_states(puzzles, indices, rng)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        groups = groups.to(self.device) if groups is not None else None
        cfg = build_solve_config(solve_dict)
        given_mask = _given_mask(x)
        with torch.no_grad():
            res = solve(model, _build_state(x, mask, groups), y, given_mask, cfg,
                        in_puzzle_mask=mask)
        solution, mask = res.solution.cpu(), mask.cpu()
        recs = [puzzles[int(i)] for i in indices]
        solutions = [solution_string(solution[j], mask[j]) for j in range(len(indices))]
        verified = [is_valid_solution(solution[j], recs[j]) for j in range(len(indices))]
        return result_rows(res, indices, cfg, solutions, verified, given_mask,
                           extra=[{"order": int(r["n"])} for r in recs])
