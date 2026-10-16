"""Benchmark name -> experiment config class, imported on demand."""

import importlib

from experiments.common import Experiment

EXPERIMENTS = {
    "sudoku": "experiments.sudoku.config:SudokuExperiment",
    "snowflake": "experiments.snowflake.config:SnowflakeExperiment",
    "maze": "experiments.maze.config:MazeExperiment",
}


def experiment_class(benchmark: str) -> type[Experiment]:
    module, _, attr = EXPERIMENTS[benchmark].partition(":")
    return getattr(importlib.import_module(module), attr)
