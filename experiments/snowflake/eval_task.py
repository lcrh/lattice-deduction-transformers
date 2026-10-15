"""Snowflake Sudoku adapter for the shared evaluator (lattice_diffusion.eval).

Split `s` is the generated parquet `snowflake_<s>.parquet` in the data
directory (e.g. `varied_test`, see data.split_name); indices are row
positions in it. Evaluation is zero-hint. Rows carry the puzzle's order so
per-order accuracy can be computed from them, and an index filter on
`orders` restricts an index list to some orders (order-transfer evals).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lattice_diffusion.eval.tasks import EvalTask

from experiments.eval_common import build_solve_config, load_eval_model, result_rows
from experiments.snowflake.data import (
    _load_puzzles, cell_to_grid_idx, load_states, puzzle_orders, split_name,
)
from experiments.snowflake.train import N_CHANNELS, VOCAB, _build_state, _given_mask
from experiments.sudoku.solve import solve


def solution_string(state: torch.Tensor, mask: torch.Tensor) -> str:
    """Digits 1..6 of the in-puzzle cells in covering-grid order, '0' where
    the cell is not a singleton."""
    vocab = state[..., :VOCAB]
    single = vocab.sum(dim=-1) == 1
    digits = ((vocab.argmax(dim=-1) + 1) * single.long())[mask]
    return "".join(str(d) for d in digits.tolist())


def is_valid_solution(state: torch.Tensor, puzzle_rec: dict) -> bool:
    """Rule check of a solved lattice against the puzzle's own topology:
    every cell a singleton, givens kept, every constraint group all-different."""
    vocab = state[..., :VOCAB]
    topology = puzzle_rec["topology"]
    digit = {}
    for cell_id, cp in topology["cell_positions"].items():
        cell = vocab[cell_to_grid_idx(cp["q"], cp["r"], cp["direction"])]
        if int(cell.sum()) != 1:
            return False
        digit[int(cell_id)] = int(cell.argmax()) + 1
    for cell_id, given in enumerate(puzzle_rec["puzzle"]):
        if given != 7 and digit[cell_id] != given:
            return False
    for group in topology["constraints"]:
        cells = group["cells"] if isinstance(group, dict) else group
        values = [digit[int(c)] for c in cells]
        if len(set(values)) != len(values):
            return False
    return True


def _uses_group_channels(model) -> bool:
    return getattr(model, "_orig_mod", model).cfg.n_channels > N_CHANNELS


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

    def filter_indices(self, split: str, indices: np.ndarray, index_filter: dict) -> np.ndarray:
        unknown = set(index_filter) - {"orders"}
        if unknown:
            raise ValueError(f"unsupported snowflake index_filter keys {sorted(unknown)}")
        orders = puzzle_orders(self._puzzles(split))
        return indices[np.isin(orders[indices], index_filter["orders"])]

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool):
        return load_eval_model(checkpoint, dropout_p, self.device, compile)

    def warmup(self, model) -> None:
        from experiments.snowflake.config import SnowflakeExperiment
        exp = SnowflakeExperiment()
        self.solve_block(model, split_name(exp.data.dataset, "test"), np.arange(8),
                         exp.solve_dict())

    def solve_block(self, model, split: str, indices: np.ndarray, solve_dict: dict) -> list[dict]:
        puzzles = self._puzzles(split)
        # Constraint-group IDs are random per load; draw them from the block's
        # seeded torch RNG so a block's rows are reproducible.
        rng = (np.random.default_rng(torch.initial_seed() % 2**63)
               if _uses_group_channels(model) else None)
        x, y, mask, groups = load_states(puzzles, indices, rng)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        groups = groups.to(self.device) if groups is not None else None
        cfg = build_solve_config(solve_dict)
        with torch.no_grad():
            res = solve(model, _build_state(x, mask, groups), y, _given_mask(x), cfg,
                        in_puzzle_mask=mask, verbose=False)
        solution, mask = res.solution.cpu(), mask.cpu()
        recs = [puzzles[int(i)] for i in indices]
        solutions = [solution_string(solution[j], mask[j]) for j in range(len(indices))]
        verified = [is_valid_solution(solution[j], recs[j]) if "constraints" in recs[j]["topology"]
                    else None for j in range(len(indices))]
        return result_rows(res, indices, cfg, solutions, verified,
                           extra=[{"order": int(r["n"])} for r in recs])
