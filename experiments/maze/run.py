"""Modal entry point for Maze: train, then evaluate with the shared evaluator.

See experiments/sudoku/run.py for how configs, overrides and checkpoint
paths work.

Usage:
    uv run modal run --detach experiments/maze/run.py --config configs/maze/paper.yaml
    uv run modal run --detach experiments/maze/run.py --config configs/maze/paper_k512.yaml
    uv run modal run --detach experiments/maze/run.py --config configs/maze/ksweep.yaml \\
        --set "data.k_solutions=64 train.seed=2"

Maze runs snapshot every `train.checkpoint_every` steps; relaunching the
same config resumes from the snapshot.
"""

import modal

from ldt.eval.modal_app import eval_app
from ldt.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)

from experiments.common import GPU, launch, run_experiment
from experiments.maze.config import MazeExperiment
from experiments.maze.train import train


app = modal.App("maze")
app.include(eval_app)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600 * 16,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(config_json: str, evaluate: bool = True) -> dict:
    return run_experiment(MazeExperiment.from_json(config_json), train,
                          CHECKPOINT_MOUNT, DATA_MOUNT, checkpoint_volume.commit,
                          evaluate=evaluate)


@app.local_entrypoint()
def entrypoint(config: str = "configs/maze/paper.yaml", set: str = "", wait: bool = True,
               evaluate: bool = True):
    launch(MazeExperiment, app, run, config, set, wait, evaluate)
