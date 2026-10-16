"""Benchmark adapters for the shared evaluator.

A task knows how to load a checkpoint for evaluation and how to solve a
block of puzzles given by absolute dataset indices. Everything else
(blocks, seeding, stopping, records) is benchmark-independent.

Tasks are registered by import path so this package does not import the
experiment code until a task is actually used.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from torch import nn

from lattice_diffusion.eval.scheduler import Row


class TaskBlock(NamedTuple):
    """Rows of one solved block and the batched forward passes it took."""

    rows: list[Row]
    model_calls: int


class EvalTask(ABC):
    """A benchmark the shared evaluator can run."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    @abstractmethod
    def split_size(self, split: str) -> int:
        """Number of puzzles in `split`."""

    @abstractmethod
    def load_model(self, checkpoint: str, dropout_p: float, compile: bool) -> nn.Module:
        """Load `checkpoint` for evaluation."""

    def filter_indices(self, split: str, indices: np.ndarray, index_filter: dict) -> np.ndarray:
        """Entries of `indices`, in order, that match `index_filter`."""
        raise NotImplementedError(f"{type(self).__name__} does not support index filters")

    @abstractmethod
    def warmup(self, model: nn.Module) -> None:
        """Run a tiny solve so torch.compile happens before timed blocks."""

    @abstractmethod
    def solve_block(self, model: nn.Module, split: str, indices: np.ndarray,
                    solve: dict) -> TaskBlock:
        """Rows for `indices` in order (the caller adds `pos` and `block`)."""


TASKS: dict[str, str] = {
    "sudoku": "experiments.sudoku.eval_task:SudokuTask",
    "snowflake": "experiments.snowflake.eval_task:SnowflakeTask",
    "maze": "experiments.maze.eval_task:MazeTask",
}


def get_task(name: str, data_dir: str) -> EvalTask:
    module, _, attr = TASKS[name].partition(":")
    return getattr(importlib.import_module(module), attr)(data_dir)
