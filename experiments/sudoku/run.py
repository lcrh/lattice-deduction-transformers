"""Modal entry point for Sudoku-Extreme: train, then evaluate with the shared evaluator.

The run is described by an experiment config (experiments/sudoku/config.py):
a YAML file plus `key.path=value` overrides. The checkpoint path is derived
from the training config, so rerunning a finished config skips training and
only (re)evaluates.

Usage:
    uv run modal run --detach experiments/sudoku/run.py --config configs/sudoku/paper.yaml
    uv run modal run --detach experiments/sudoku/run.py --config configs/sudoku/paper.yaml \\
        --set "train.steps=2000 train.seed=1 eval.stop=full"
"""

import modal

from lattice_diffusion.eval.modal_app import eval_app
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)

from experiments.common import GPU, launch, run_experiment
from experiments.sudoku.config import SudokuExperiment
from experiments.sudoku.train import train


app = modal.App("sudoku")
app.include(eval_app)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600 * 4,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(config_json: str, evaluate: bool = True) -> dict:
    return run_experiment(SudokuExperiment.from_json(config_json), train,
                          CHECKPOINT_MOUNT, DATA_MOUNT, checkpoint_volume.commit,
                          evaluate=evaluate)


@app.local_entrypoint()
def entrypoint(config: str = "configs/sudoku/paper.yaml", set: str = "", wait: bool = True,
               evaluate: bool = True):
    launch(SudokuExperiment, app, run, config, set, wait, evaluate)
