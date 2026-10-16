"""Snowflake Sudoku experiment config.

Defaults (configs/snowflake/paper.yaml): the varied-topology dataset, trained
and evaluated on orders 4-8. Order-transfer configs are configs/snowflake/ood_*.yaml.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from lattice_diffusion.eval.config import EvalConfig
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment, check_options, with_step
from experiments.snowflake.data import GRID_COLS, GRID_ROWS, SEQ_LEN, VOCAB, SnowflakeConfig
from experiments.snowflake.train import TrainConfig, n_channels, snowflake_step
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
    data: SnowflakeConfig = field(default_factory=lambda: SnowflakeConfig(
        dataset="varied", n_puzzles=500, train_orders=[4, 5, 6, 7, 8], translate_aug=True))
    train: TrainConfig = field(default_factory=TrainConfig)
    solve: SolveConfig = field(default_factory=lambda: SolveConfig(
        step=snowflake_step(cls_threshold=0.6, augment=True),
        max_rounds=1000, n_chains=64, batch_size=512))
    eval: EvalConfig = field(default_factory=lambda: EvalConfig(
        index_list="snowflake_varied_test", index_filter={"orders": [4, 5, 6, 7, 8]},
        stop="full", workers=8))

    def __post_init__(self):
        self.model = dataclasses.replace(self.model,
                                         n_channels=n_channels(self.data.constraint_group_embed),
                                         seq_len=SEQ_LEN,
                                         grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
        self.train = with_step(self.train, vocab_dim=VOCAB, augment_dihedral=False)
        self.solve = with_step(self.solve, vocab_dim=VOCAB, augment_dihedral=False)
        check_options(self.model, self.train, self.solve, carry_supported=False)
