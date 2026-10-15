"""Modal entry point for Snowflake Sudoku: train, then evaluate with the shared evaluator.

See experiments/sudoku/run.py for how configs, overrides and checkpoint
paths work.

Usage:
    uv run modal run --detach experiments/snowflake/run.py --config configs/snowflake/paper.yaml
"""

import modal

from lattice_diffusion.eval.modal_app import eval_app
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)

from experiments.common import launch, run_experiment
from experiments.snowflake.config import SnowflakeExperiment
from experiments.snowflake.train import train


app = modal.App("snowflake")
app.include(eval_app)


@app.function(
    image=image,
    gpu="B200",
    timeout=3600 * 4,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(config_json: str, evaluate: bool = True) -> dict:
    return run_experiment(SnowflakeExperiment.from_json(config_json), train,
                          CHECKPOINT_MOUNT, DATA_MOUNT, checkpoint_volume.commit,
                          evaluate=evaluate)


@app.local_entrypoint()
def entrypoint(config: str = "configs/snowflake/paper.yaml", set: str = "", wait: bool = True,
               evaluate: bool = True):
    launch(SnowflakeExperiment, app, run, config, set, wait, evaluate)
