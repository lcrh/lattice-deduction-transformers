"""Sudoku-Extreme experiment config. Defaults are the paper recipe
(configs/sudoku/paper.yaml spells them out)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig
from lattice_diffusion.eval.config import EvalConfig
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment
from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig
from experiments.sudoku.train import TrainConfig


def check_cls_head(model: LoopedTransformerConfig, train) -> None:
    """The conflict head exists iff it is trained."""
    if model.cls_token != (train.conflict_loss_weight > 0):
        raise ValueError(
            f"model.cls_token={model.cls_token} but train.conflict_loss_weight="
            f"{train.conflict_loss_weight}; the conflict head is trained iff it exists")


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
        check_cls_head(self.model, self.train)
        if self.data.split != "train":
            raise ValueError("data.split must be train; evaluation splits live in eval")

    def localized(self, data_dir: str) -> "SudokuExperiment":
        return dataclasses.replace(self, data=dataclasses.replace(self.data, cache_dir=data_dir))
