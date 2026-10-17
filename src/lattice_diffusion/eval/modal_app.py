"""Modal deployment of the shared evaluator.

`coordinate` (CPU) owns the run directory: it schedules blocks, writes block
files and the summary, and commits the volume. `BlockWorker` (B200) loads the
model once per container, then solves blocks.

Workers are parametrized only by what determines the loaded model, so a warm
container serves every block of an eval and every eval of the same model.
"""

import threading
import time
from collections import Counter
from typing import TypedDict

import modal
import numpy as np

from lattice_diffusion.eval.records import EvalSpec, format_summary
from lattice_diffusion.eval.runner import BlockArgs, run_eval, solve_block
from lattice_diffusion.eval.scheduler import Row, ThreadExecutor
from lattice_diffusion.eval.tasks import get_task
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT, checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.provenance import header

EVAL_ROOT = f"{CHECKPOINT_MOUNT}/eval"

eval_app = modal.App("lattice-eval")


class BlockResult(TypedDict):
    rows: list[Row]
    model_calls: int            # batched forward passes
    seconds: float              # time solving the block
    startup_seconds: float      # container load/compile time, on its first block only


@eval_app.cls(
    image=image, gpu="B200", timeout=6 * 3600, secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
    max_containers=100, scaledown_window=10,
    # Blocks are seeded and idempotent, so a failed or preempted call is safe to rerun.
    retries=modal.Retries(max_retries=3, initial_delay=5.0, backoff_coefficient=2.0),
)
class BlockWorker:
    checkpoint: str = modal.parameter()
    task: str = modal.parameter()
    dropout_p: str = modal.parameter()     # float as str: Modal has no float parameters
    compile: bool = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        t0 = time.time()
        self.task_impl = get_task(self.task, DATA_MOUNT)
        self.model = self.task_impl.load_model(self.checkpoint, float(self.dropout_p),
                                               self.compile)
        self.warm = not self.compile
        self.startup_seconds = time.time() - t0
        print(f"[worker] loaded {self.checkpoint} ({self.task}, compile={self.compile}) "
              f"in {self.startup_seconds:.1f}s", flush=True)

    @modal.method()
    def solve(self, args: BlockArgs) -> BlockResult:
        """Solve one block; the container's first call also reports its startup time."""
        if not self.warm:
            # Compile on the first block's own data: grid sizes and solver settings vary by spec.
            t0 = time.time()
            self.task_impl.warmup(self.model, args["split"],
                                  np.asarray(args["indices"], dtype=np.int64), args["solve"])
            self.startup_seconds += time.time() - t0
            self.warm = True
        print(f"[worker] block {args['block']}: {len(args['indices'])} puzzles "
              f"from {args['split']}", flush=True)
        t0 = time.time()
        out = solve_block(self.task_impl, self.model, args)
        seconds = time.time() - t0
        counts = Counter(row["status"] for row in out.rows)
        print(f"[worker] block {args['block']} done: correct={counts['correct']} "
              f"wrong={counts['wrong']} timeout={counts['timeout']} "
              f"calls={out.model_calls} {seconds:.1f}s", flush=True)
        startup, self.startup_seconds = self.startup_seconds, 0.0
        return BlockResult(rows=out.rows, model_calls=out.model_calls,
                           seconds=seconds, startup_seconds=startup)


@eval_app.function(
    image=image, cpu=2, memory=4096, timeout=24 * 3600, secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def coordinate(spec_json: str) -> dict:
    spec = EvalSpec.from_json(spec_json)
    print(header("eval", exp_id=spec.exp_id, entry=spec.entry, spec_hash=spec.spec_hash(),
                 code=spec.code_sha, checkpoint=spec.checkpoint,
                 run_dir=spec.run_dir(EVAL_ROOT)), flush=True)
    checkpoint_volume.reload()
    worker = BlockWorker.with_options(
        max_containers=spec.n_workers, scaledown_window=spec.keep_warm_s,
    )(
        checkpoint=spec.checkpoint, task=spec.task,
        dropout_p=repr(float(spec.solve.get("dropout_p", 0.0))), compile=spec.compile,
    )
    meta: dict[int, dict] = {}
    lock = threading.Lock()

    def make_executor(block_args):
        # One blocking remote call per in-flight block; the scheduler bounds how many.
        def solve(block: int) -> list[Row]:
            out = worker.solve.remote(block_args(block))
            with lock:
                meta[block] = {"model_calls": out["model_calls"], "seconds": out["seconds"],
                               "startup_seconds": out["startup_seconds"]}
            return out["rows"]

        return ThreadExecutor(solve, n_workers=spec.window)

    task = get_task(spec.task, DATA_MOUNT)
    summary = run_eval(
        spec, EVAL_ROOT, make_executor,
        persist=checkpoint_volume.commit,
        block_meta=lambda block: meta.pop(block, None),
        split_size=task.split_size,
        filter_indices=task.filter_indices if spec.index_filter is not None else None,
    )
    print(format_summary(summary), flush=True)
    return {"run_dir": summary.run_dir, "summary": format_summary(summary)}
