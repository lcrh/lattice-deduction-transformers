"""Sudoku-Extreme experiment config.

The defaults are spelled out in configs/sudoku/paper.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ldt.data.sudoku_extreme import SudokuExtremeConfig
from ldt.eval.config import EvalConfig
from ldt.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment, check_options
from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig
from experiments.sudoku.train import TrainConfig


@dataclass
class SudokuExperiment(Experiment):
    BENCHMARK = "sudoku"
    ENVIRONMENT = ("data.cache_dir",)

    model: LoopedTransformerConfig = field(
        default_factory=lambda: LoopedTransformerConfig(cls_token=True))
    data: SudokuExtremeConfig = field(
        default_factory=lambda: SudokuExtremeConfig(n_puzzles=1000, seed=42))
    # No per-step augmentation in training: dataset-level augmentation is enough.
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(step=StepConfig(augment=False)))
    solve: SolveConfig = field(default_factory=lambda: SolveConfig(
        step=StepConfig(cls_threshold=0.6, augment=True, digit_policy="argmax"),
        max_rounds=1000, n_chains=64, batch_size=512))
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        check_options(self.model, self.train, self.solve)
        if self.data.split != "train":
            raise ValueError("data.split must be train; evaluation splits live in eval")
