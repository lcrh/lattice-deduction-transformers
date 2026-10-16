"""Three-step training runs on synthetic data, one per benchmark.

Each returns the trained model's state_dict. The settings match the ones
the reference fixtures were captured with; `train_kw` overrides TrainConfig
fields (e.g. to switch on monitoring).
"""

from pathlib import Path

import torch

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig
from experiments.sudoku.dpll import StepConfig
from tests.equivalence import fakes


def _state(path) -> dict:
    return torch.load(path, weights_only=False)["model_state_dict"]


def sudoku(tmp: Path, **train_kw) -> dict:
    from experiments.sudoku.config import SudokuExperiment
    from experiments.sudoku.train import TrainConfig, train

    fakes.write_sudoku_caches(tmp)
    exp = SudokuExperiment(
        model=fakes.tiny_model_cfg(),
        data=SudokuExtremeConfig(cache_dir=str(tmp), seed=42, n_puzzles=100),
        train=TrainConfig(**{**dict(steps=3, batch_size=16, seed=0, step=StepConfig(augment=True),
                                    compile=False, monitor_every=0, log_every=1), **train_kw}),
    )
    with fakes.small_sudoku_val():
        train(exp, tmp / "sudoku.pt")
    return _state(tmp / "sudoku.pt")


def snowflake(tmp: Path, **train_kw) -> dict:
    from experiments.snowflake.config import SnowflakeExperiment
    from experiments.snowflake.data import SnowflakeConfig
    from experiments.snowflake.train import TrainConfig, snowflake_step, train

    fakes.write_snowflake_parquet(tmp / "snowflake_train.parquet", 60, seed=31)
    fakes.write_snowflake_parquet(tmp / "snowflake_test.parquet", 10, seed=32)
    exp = SnowflakeExperiment(
        model=fakes.tiny_model_cfg(),
        data=SnowflakeConfig(data_dir=str(tmp), seed=42, n_puzzles=40),
        train=TrainConfig(**{**dict(steps=3, batch_size=16, seed=0,
                                    step=snowflake_step(augment=True), compile=False,
                                    monitor_every=0, log_every=1), **train_kw}),
    )
    train(exp, tmp / "snowflake.pt")
    return _state(tmp / "snowflake.pt")


def maze(tmp: Path, **train_kw) -> dict:
    from experiments.maze.config import MazeExperiment
    from experiments.maze.data import MazeConfig
    from experiments.maze.train import TrainConfig, maze_step, train

    exp = MazeExperiment(
        model=fakes.tiny_model_cfg(),
        data=MazeConfig(dataset="synthetic", cache_dir=str(tmp), seed=42, grid_size=7,
                        hard=False, n_puzzles=40, k_solutions=2),
        train=TrainConfig(**{**dict(steps=3, batch_size=16, seed=0,
                                    step=maze_step(threshold=0.5, augment=True),
                                    pool_size_mult=2.0, compile=False, monitor_every=0,
                                    log_every=1, checkpoint_every=0), **train_kw}),
    )
    train(exp, tmp / "maze.pt")
    return _state(tmp / "maze.pt")


RUNS = {"sudoku": sudoku, "snowflake": snowflake, "maze": maze}
