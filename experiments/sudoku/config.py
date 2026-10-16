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


def check_options(model: LoopedTransformerConfig, train, solve: SolveConfig,
                  carry_supported: bool = True) -> None:
    """Cross-block constraints between model, training and solver options."""
    if model.cls_token != (train.conflict_loss_weight > 0):
        raise ValueError(
            f"model.cls_token={model.cls_token} but train.conflict_loss_weight="
            f"{train.conflict_loss_weight}; the conflict head is trained iff it exists")
    if train.supervise not in ("all", "final"):
        raise ValueError(f"train.supervise must be all or final, got {train.supervise!r}")
    if model.carry != "off":
        if not carry_supported:
            raise ValueError("latent carry is only implemented for the Sudoku trainer")
        # The carried latent lives in a fixed frame and belongs to one
        # trajectory: no per-step augmentation, one pass per step, and
        # conflicts restart from the root (where the latent is reset).
        for where, step in (("train.step", train.step), ("solve.step", solve.step)):
            if step.augment or step.max_passes != 1:
                raise ValueError(f"model.carry={model.carry!r} needs {where}.augment=false "
                                 f"and {where}.deduce_passes=1")
        if solve.backtrack != "root":
            raise ValueError(f"model.carry={model.carry!r} needs solve.backtrack=root")


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

    def localized(self, data_dir: str) -> "SudokuExperiment":
        return dataclasses.replace(self, data=dataclasses.replace(self.data, cache_dir=data_dir))
