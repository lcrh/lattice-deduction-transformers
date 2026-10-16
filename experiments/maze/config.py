"""Maze experiment config.

Defaults: Maze-Hard with one target path (configs/maze/paper.yaml);
configs/maze/paper_k512.yaml uses 512 paths and configs/maze/ksweep.yaml the
15x15 synthetic K sweep.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from lattice_diffusion.data.maze_hard import N_CHANNELS
from lattice_diffusion.eval.config import EvalConfig
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig

from experiments.common import Experiment, check_options, with_step
from experiments.maze.data import MazeConfig, grid_dims
from experiments.maze.train import TrainConfig, maze_step
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

    model: LoopedTransformerConfig = field(default_factory=lambda: LoopedTransformerConfig(
        dim=192, use_rope=True, cls_token=True))
    data: MazeConfig = field(default_factory=lambda: MazeConfig(dataset="maze_hard", seed=42))
    train: TrainConfig = field(default_factory=TrainConfig)
    solve: SolveConfig = field(default_factory=lambda: SolveConfig(
        step=maze_step(threshold=0.5, cls_threshold=0.53, augment=True),
        max_rounds=1000, n_chains=64, batch_size=64))
    eval: EvalConfig = field(default_factory=lambda: EvalConfig(
        index_list="maze_test", stop="full", dropout_p=0.0, block_size=50, workers=20,
        compile=True))

    def __post_init__(self):
        h, w = grid_dims(self.data)
        self.model = dataclasses.replace(self.model, n_channels=N_CHANNELS, seq_len=h * w,
                                         grid_rows=h, grid_cols=w)
        self.train = with_step(self.train, augment_dihedral=True, permute_digits=False)
        self.solve = with_step(self.solve, augment_dihedral=True, permute_digits=False)
        check_options(self.model, self.train, self.solve, carry_supported=False)
        if self.data.split != "train":
            raise ValueError("data.split must be train; evaluation splits live in eval")

    def solve_dict(self) -> dict:
        """Solver settings plus the maze description the evaluator needs for synthetic pools."""
        d = self.data
        return {**super().solve_dict(), "maze_data": {
            "dataset": d.dataset, "grid_size": d.grid_size, "hard": d.hard,
            "wall_frac_lo": d.wall_frac_lo, "wall_frac_hi": d.wall_frac_hi,
            "n_puzzles": self.eval.n,
        }}
