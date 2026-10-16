"""Benchmark name -> experiment config class, imported on demand."""

import importlib

EXPERIMENTS = {
    "sudoku": "experiments.sudoku.config:SudokuExperiment",
    "snowflake": "experiments.snowflake.config:SnowflakeExperiment",
    "maze": "experiments.maze.config:MazeExperiment",
}


def experiment_class(benchmark: str):
    module, _, attr = EXPERIMENTS[benchmark].partition(":")
    return getattr(importlib.import_module(module), attr)
