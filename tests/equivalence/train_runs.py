"""Three-step training runs on synthetic data, one per benchmark.

Each returns the trained model's state_dict. The settings match the ones
the reference fixtures were captured with.
"""

from pathlib import Path

import torch

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig
from experiments.sudoku.dpll import StepConfig
from tests.equivalence import fakes


def _state(path) -> dict:
    return torch.load(path, weights_only=False)["model_state_dict"]


def sudoku(tmp: Path) -> dict:
    from experiments.sudoku.train import TrainConfig, train

    fakes.write_sudoku_caches(tmp)
    with fakes.small_sudoku_val():
        path = train(TrainConfig(
            steps=3, batch_size=16, seed=0, weight_decay=0.1, conflict_loss_weight=0.1,
            step=StepConfig(augment=True), compile=False, eval_every=0, log_every=1,
            model=fakes.tiny_model_cfg(),
            data=SudokuExtremeConfig(cache_dir=str(tmp), batch_size=16, seed=42, n_puzzles=100),
            out_dir=str(tmp / "ck"), name="sudoku",
        ))
    return _state(path)


def snowflake(tmp: Path) -> dict:
    from experiments.snowflake.data import SnowflakeConfig
    from experiments.snowflake.train import TrainConfig, train

    fakes.write_snowflake_parquet(tmp / "snowflake_train.parquet", 60, seed=31)
    fakes.write_snowflake_parquet(tmp / "snowflake_test.parquet", 10, seed=32)
    path = train(TrainConfig(
        steps=3, batch_size=16, seed=0, weight_decay=0.1, conflict_loss_weight=0.1,
        step=StepConfig(augment=True, augment_dihedral=False, vocab_dim=6),
        compile=False, eval_every=0, log_every=1,
        model=fakes.tiny_model_cfg(n_channels=7, seq_len=150, grid_rows=15, grid_cols=10),
        data=SnowflakeConfig(data_path=str(tmp / "snowflake_train.parquet"),
                             batch_size=16, seed=42, n_puzzles=40),
        eval_data=SnowflakeConfig(data_path=str(tmp / "snowflake_test.parquet"),
                                  n_puzzles=200, batch_size=200, seed=200,
                                  zero_hint_weight=1.0, correct_hint_weight=0.0,
                                  error_hint_weight=0.0),
        out_dir=str(tmp / "ck"), name="snowflake",
    ))
    return _state(path)


def maze(tmp: Path) -> dict:
    from experiments.maze.data import MazeConfig
    from experiments.maze.train import TrainConfig, train

    path = train(TrainConfig(
        steps=3, batch_size=16, seed=0, weight_decay=0.1, conflict_loss_weight=0.1,
        step=StepConfig(augment=True, augment_dihedral=True, permute_digits=False,
                        threshold=0.5),
        pool_size_mult=2.0, compile=False, eval_every=0, log_every=1, eval_n_puzzles=4,
        model=fakes.tiny_model_cfg(n_channels=5, seq_len=49, grid_rows=7, grid_cols=7),
        data=MazeConfig(dataset="synthetic", cache_dir=str(tmp), batch_size=16, seed=42,
                        grid_size=7, hard=False, n_puzzles=40, k_solutions=2),
        out_dir=str(tmp / "ck"), name="maze",
    ))
    return _state(path)


RUNS = {"sudoku": sudoku, "snowflake": snowflake, "maze": maze}
