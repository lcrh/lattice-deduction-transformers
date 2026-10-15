"""Modal deployment of the shared evaluator.

`coordinate` runs on a CPU container and owns the run directory: it
schedules blocks, writes block files and the summary, and commits the
volume. `BlockWorker` runs on B200s; each container loads (and optionally
compiles) the model once, then solves blocks. Benchmark entry scripts
include `eval_app` into their own app and call
`coordinate.remote(spec.to_json())`.

Workers are parametrized only by what determines the loaded model
(checkpoint, task, eval dropout, compile). The solver config, eval seed and
block are passed per call, so containers are reused across blocks of one
eval and across evals of the same model (seeds, presets, solver sweeps).
`keep_warm_s` keeps idle containers alive, e.g. while a slow block holds
the scheduling window.
"""

import threading
import time

import modal

from lattice_diffusion.eval.records import EvalSpec, format_summary
from lattice_diffusion.eval.runner import run_eval, solve_block
from lattice_diffusion.eval.scheduler import ThreadExecutor
from lattice_diffusion.eval.tasks import get_task
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT, checkpoint_volume, data_volume, hf_secret, image,
)

EVAL_ROOT = f"{CHECKPOINT_MOUNT}/eval"

eval_app = modal.App("lattice-eval")


@eval_app.cls(
    image=image, gpu="B200", timeout=6 * 3600, secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    max_containers=100, scaledown_window=300,
)
class BlockWorker:
    checkpoint: str = modal.parameter()
    task: str = modal.parameter()
    dropout_p: str = modal.parameter()     # float as str: Modal has no float parameters
    compile: bool = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        t0 = time.time()
        self._task = get_task(self.task, DATA_MOUNT)
        self.model = self._task.load_model(self.checkpoint, float(self.dropout_p),
                                           self.compile)
        if self.compile:
            self._task.warmup(self.model)
        self.startup_seconds = time.time() - t0

    @modal.method()
    def solve(self, args: dict) -> dict:
        """Solve one block. The first call on each container also reports
        that container's load/compile time."""
        t0 = time.time()
        rows = solve_block(self._task, self.model, args)
        startup, self.startup_seconds = self.startup_seconds, 0.0
        return {"rows": rows, "seconds": time.time() - t0, "startup_seconds": startup}


@eval_app.function(
    image=image, cpu=2, memory=4096, timeout=24 * 3600, secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def coordinate(spec_json: str) -> dict:
    spec = EvalSpec.from_json(spec_json)
    checkpoint_volume.reload()
    worker = BlockWorker.with_options(
        max_containers=spec.n_workers, scaledown_window=spec.keep_warm_s,
    )(
        checkpoint=spec.checkpoint, task=spec.task,
        dropout_p=repr(float(spec.solve.get("dropout_p", 0.0))), compile=spec.compile,
    )
    seconds = {"blocks": 0.0, "startup": 0.0}
    lock = threading.Lock()

    def make_executor(block_args):
        # One blocking remote call per in-flight block; the scheduler keeps
        # at most `window` blocks in flight.
        def solve(block: int) -> list:
            out = worker.solve.remote(block_args(block))
            with lock:
                seconds["blocks"] += out["seconds"]
                seconds["startup"] += out["startup_seconds"]
            return out["rows"]

        return ThreadExecutor(solve, n_workers=spec.window)

    summary = run_eval(
        spec, EVAL_ROOT, make_executor,
        persist=checkpoint_volume.commit,
        block_seconds=lambda: seconds["blocks"],
        startup_seconds=lambda: seconds["startup"],
        split_size=lambda split: get_task(spec.task, DATA_MOUNT).split_size(split),
    )
    print(format_summary(summary), flush=True)
    return {"run_dir": summary.run_dir, "summary": format_summary(summary)}
