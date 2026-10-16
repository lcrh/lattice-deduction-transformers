"""Snowflake Sudoku experiment config. Defaults are the paper recipe
(configs/snowflake/paper.yaml spells them out)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from lattice_diffusion.eval.config import EvalConfig
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment
from experiments.snowflake.data import SnowflakeConfig
from experiments.snowflake.train import (
    GRID_COLS, GRID_ROWS, N_CHANNELS, SEQ_LEN, VOCAB, TrainConfig, snowflake_step,
)
from experiments.sudoku.config import check_cls_head
from experiments.sudoku.solve import SolveConfig


@dataclass
class SnowflakeExperiment(Experiment):
    BENCHMARK = "snowflake"
    # Fixed by the covering-grid encoding (data.py).
    DERIVED = ("model.n_channels", "model.seq_len", "model.grid_rows", "model.grid_cols",
               "train.step.vocab_dim", "train.step.augment_dihedral",
               "solve.step.vocab_dim", "solve.step.augment_dihedral")
    ENVIRONMENT = ("data.data_dir",)

    name: str = "paper"
    model: LoopedTransformerConfig = field(
        default_factory=lambda: LoopedTransformerConfig(cls_token=True))
    data: SnowflakeConfig = field(default_factory=lambda: SnowflakeConfig(n_puzzles=500))
    train: TrainConfig = field(default_factory=TrainConfig)
    solve: SolveConfig = field(default_factory=lambda: SolveConfig(
        step=snowflake_step(cls_threshold=0.6, augment=True),
        max_rounds=1000, n_chains=64, batch_size=512))
    eval: EvalConfig = field(default_factory=lambda: EvalConfig(
        index_list="snowflake_test", stop="full", workers=8))

    def __post_init__(self):
        self.model = dataclasses.replace(self.model, n_channels=N_CHANNELS, seq_len=SEQ_LEN,
                                         grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
        self.train = dataclasses.replace(self.train, step=dataclasses.replace(
            self.train.step, vocab_dim=VOCAB, augment_dihedral=False))
        self.solve = dataclasses.replace(self.solve, step=dataclasses.replace(
            self.solve.step, vocab_dim=VOCAB, augment_dihedral=False))
        check_cls_head(self.model, self.train)

    def localized(self, data_dir: str) -> "SnowflakeExperiment":
        return dataclasses.replace(self, data=dataclasses.replace(self.data, data_dir=data_dir))
