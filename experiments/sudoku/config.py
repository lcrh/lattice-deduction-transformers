"""Sudoku-Extreme experiment config.

The defaults are spelled out in configs/sudoku/paper.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig
from lattice_diffusion.eval.config import EvalConfig
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment, check_options
from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig
from experiments.sudoku.train import TrainConfig


@dataclass
class SudokuExperiment(Experiment):
    BENCHMARK = "sudoku"
    ENVIRONMENT = ("data.cache_dir",)

    name: str = "paper"
    model: LoopedTransformerConfig = field(
        default_factory=lambda: LoopedTransformerConfig(cls_token=True))
    data: SudokuExtremeConfig = field(
        default_factory=lambda: SudokuExtremeConfig(n_puzzles=1000, seed=42))
    train: TrainConfig = field(default_factory=TrainConfig)
    solve: SolveConfig = field(default_factory=lambda: SolveConfig(
        step=StepConfig(cls_threshold=0.6, augment=True),
        max_rounds=1000, n_chains=64, batch_size=512))
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        check_options(self.model, self.train, self.solve)
        if self.data.split != "train":
            raise ValueError("data.split must be train; evaluation splits live in eval")
