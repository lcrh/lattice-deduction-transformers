"""Maze experiment config. Defaults are the paper's Maze-Hard K=1 recipe
(configs/maze/paper.yaml); configs/maze/paper_k512.yaml and
configs/maze/ksweep.yaml cover Table 3's K=512 row and Figure 3."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from lattice_diffusion.data.maze_hard import N_CHANNELS
from lattice_diffusion.eval.config import EvalConfig
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment
from experiments.maze.data import MazeConfig, grid_dims
from experiments.maze.train import TrainConfig, maze_step
from experiments.sudoku.config import check_cls_head
from experiments.sudoku.solve import SolveConfig


@dataclass
class MazeExperiment(Experiment):
    BENCHMARK = "maze"
    # Grid shape comes from the dataset; the channel semantics fix the
    # augmentation (dihedral only, no channel permutation).
    DERIVED = ("model.n_channels", "model.seq_len", "model.grid_rows", "model.grid_cols",
               "train.step.augment_dihedral", "train.step.permute_digits",
               "solve.step.augment_dihedral", "solve.step.permute_digits")
    ENVIRONMENT = ("data.cache_dir",)

    name: str = "paper"
    model: LoopedTransformerConfig = field(default_factory=lambda: LoopedTransformerConfig(
        dim=192, use_rope=True, cls_token=True))
    data: MazeConfig = field(default_factory=lambda: MazeConfig(dataset="maze_hard", seed=42))
    train: TrainConfig = field(default_factory=TrainConfig)
    solve: SolveConfig = field(default_factory=lambda: SolveConfig(
        step=maze_step(threshold=0.5, cls_threshold=0.53, augment=True),
        max_rounds=1000, n_chains=64, batch_size=64))
    eval: EvalConfig = field(default_factory=lambda: EvalConfig(
        index_list="maze_test", stop="full", dropout_p=0.0, block_size=50, workers=20))

    def __post_init__(self):
        h, w = grid_dims(self.data)
        self.model = dataclasses.replace(self.model, n_channels=N_CHANNELS, seq_len=h * w,
                                         grid_rows=h, grid_cols=w)
        self.train = dataclasses.replace(self.train, step=dataclasses.replace(
            self.train.step, augment_dihedral=True, permute_digits=False))
        self.solve = dataclasses.replace(self.solve, step=dataclasses.replace(
            self.solve.step, augment_dihedral=True, permute_digits=False))
        check_cls_head(self.model, self.train)
        if self.data.split != "train":
            raise ValueError("data.split must be train; evaluation splits live in eval")

    def localized(self, data_dir: str) -> "MazeExperiment":
        return dataclasses.replace(self, data=dataclasses.replace(self.data, cache_dir=data_dir))

    def solve_dict(self) -> dict:
        """The evaluator also needs the dataset to build synthetic eval pools
        and to size the grid for path validation."""
        d = self.data
        return {**super().solve_dict(), "maze_data": {
            "dataset": d.dataset, "grid_size": d.grid_size, "hard": d.hard,
            "wall_frac_lo": d.wall_frac_lo, "wall_frac_hi": d.wall_frac_hi,
            "n_puzzles": self.eval.n,
        }}
