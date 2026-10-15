"""Benchmark adapters for the shared evaluator.

A task knows how to load a checkpoint for evaluation and how to solve a
block of puzzles given by absolute dataset indices. Everything else
(blocks, seeding, stopping, records) is benchmark-independent.

Tasks are registered by import path so this package does not import the
experiment code until a task is actually used.
"""

from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np

from lattice_diffusion.eval.scheduler import Row


class EvalTask(Protocol):
    name: str

    def __init__(self, data_dir: str) -> None: ...

    def split_size(self, split: str) -> int: ...

    def default_solve(self) -> dict: ...

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool): ...

    def warmup(self, model) -> None:
        """Run a tiny solve so torch.compile happens before timed blocks."""
        ...

    def solve_block(self, model, split: str, indices: np.ndarray, solve: dict) -> list[Row]:
        """Rows for `indices` in order, with every field in records.py except
        `pos` and `block` (added by the caller)."""
        ...


TASKS: dict[str, str] = {
    "sudoku": "experiments.sudoku.eval_task:SudokuTask",
}


def get_task(name: str, data_dir: str) -> EvalTask:
    module, _, attr = TASKS[name].partition(":")
    return getattr(importlib.import_module(module), attr)(data_dir)
