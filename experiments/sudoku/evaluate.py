"""Evaluate a Sudoku-Extreme checkpoint with the shared parallel evaluator.

Walks an index list in blocks across B200 workers. With the `sequential`
preset it stops once the error rate is pinned down (see
lattice_diffusion/eval/stop.py); with `full` it evaluates every index.
Results land on the checkpoint volume under /checkpoints/eval/<ckpt stem>/;
rerunning the same command resumes from the blocks already written.

Usage:
    # Sequential eval on the full test split (stops early for weak models)
    uv run modal run --detach experiments/sudoku/evaluate.py \
      --checkpoint /checkpoints/sudoku/<name>.pt

    # Every test puzzle
    uv run modal run --detach experiments/sudoku/evaluate.py \
      --checkpoint /checkpoints/sudoku/<name>.pt --stop full --workers 100

    # Explicit puzzle list (e.g. a portfolio failure set), one index per line
    uv run modal run --detach experiments/sudoku/evaluate.py \
      --checkpoint /checkpoints/sudoku/<name>.pt \
      --indices-file hard.txt --split test --stop full --max-rounds 50000
"""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import modal

from lattice_diffusion.eval.modal_app import coordinate, eval_app

from experiments.sudoku.eval_task import make_eval_spec


app = modal.App("sudoku-evaluate")
app.include(eval_app)


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    index_list: str = "sudoku_test_perm",
    stop: Literal["sequential", "full"] = "sequential",
    workers: int = 32,
    max_in_flight: int = 0,
    keep_warm_s: int = 300,
    block_size: int = 512,
    eval_seed: int = 0,
    tag: str = "",
    compile: bool = False,
    threshold: float | None = None,
    temp_decide: float | None = None,
    cls_threshold: float | None = None,
    augment: bool | None = None,
    n_chains: int | None = None,
    batch_size: int | None = None,
    max_rounds: int | None = None,
    dropout_p: float | None = None,
    estimate_sequential: bool | None = None,
    seq_drain_max_rounds: int | None = None,
    indices_file: str = "",
    split: str = "test",
):
    explicit = None
    if indices_file:
        explicit = [int(line) for line in Path(indices_file).read_text().split() if line]
    spec = make_eval_spec(
        checkpoint, index_list=index_list, stop=stop, workers=workers,
        max_in_flight=max_in_flight, keep_warm_s=keep_warm_s, block_size=block_size,
        eval_seed=eval_seed, tag=tag, compile=compile, threshold=threshold, temp_decide=temp_decide,
        cls_threshold=cls_threshold, augment=augment, n_chains=n_chains,
        batch_size=batch_size, max_rounds=max_rounds, dropout_p=dropout_p,
        estimate_sequential=estimate_sequential, seq_drain_max_rounds=seq_drain_max_rounds,
        explicit_indices=explicit, split=split,
    )
    print(f"Eval spec: {asdict(spec) | {'explicit_indices': len(explicit or [])}}", flush=True)
    result = coordinate.remote(spec.to_json())
    print(f"\n{result['run_dir']}\n{result['summary']}", flush=True)
