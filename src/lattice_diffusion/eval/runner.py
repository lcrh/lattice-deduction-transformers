"""Backend-independent eval driver.

`run_eval` resolves the index list, resumes from blocks already on disk,
schedules the rest through an executor, and writes the summary.
`solve_block` is what a worker runs for one block. Both are used by the
Modal app (modal_app.py) and by local runs / tests with a thread executor.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from lattice_diffusion.eval import indices as index_lists
from lattice_diffusion.eval.records import (
    EvalSpec, Summary, check_or_write_spec, read_blocks, summarize,
    write_block, write_summary,
)
from lattice_diffusion.eval.scheduler import Executor, Row, block_bounds, run_blocks
from lattice_diffusion.eval.stop import PrefixCounts, stop_from_dict
from lattice_diffusion.eval.tasks import EvalTask


def resolve_index_list(spec: EvalSpec) -> index_lists.IndexList:
    if spec.index_list == "explicit":
        if spec.explicit_indices is None or spec.explicit_split is None:
            raise ValueError("explicit index list needs explicit_indices and explicit_split")
        return index_lists.explicit(spec.task, spec.explicit_split, spec.explicit_indices)
    lst = index_lists.load(spec.index_list)
    if lst.task != spec.task:
        raise ValueError(f"index list {spec.index_list} is for task {lst.task}, not {spec.task}")
    return lst


def block_seed(eval_seed: int, block: int) -> int:
    digest = hashlib.sha256(f"{eval_seed}:{block}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def block_args(spec: EvalSpec, lst: index_lists.IndexList, block: int) -> dict:
    """Everything a worker needs for one block; plain types for RPC."""
    start, end = block_bounds(block, spec.block_size, len(lst))
    expected = (index_lists.SPECS[lst.name].split_size
                if lst.name in index_lists.SPECS else None)
    return {
        "solve": spec.solve,
        "eval_seed": spec.eval_seed,
        "block": block,
        "start": start,
        "split": lst.split,
        "split_size": expected,
        "indices": lst.indices[start:end].tolist(),
    }


def solve_block(task: EvalTask, model, args: dict) -> list[Row]:
    """Seed from (eval_seed, block), solve, and tag rows with pos/block.

    The seed depends only on the block id, so a block's rows do not depend
    on which worker ran it or what it ran before.
    """
    if args["split_size"] is not None and task.split_size(args["split"]) != args["split_size"]:
        raise ValueError(f"{args['split']} split has {task.split_size(args['split'])} items, "
                         f"index list expects {args['split_size']}")
    seed = block_seed(args["eval_seed"], args["block"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rows = task.solve_block(model, args["split"], np.asarray(args["indices"], dtype=np.int64),
                            args["solve"])
    for k, row in enumerate(rows):
        row["pos"] = args["start"] + k
        row["block"] = args["block"]
    return rows


def run_eval(
    spec: EvalSpec,
    root: str | Path,
    make_executor: Callable[[Callable[[int], dict]], Executor],
    persist: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
    block_seconds: Callable[[], float] = lambda: 0.0,
    startup_seconds: Callable[[], float] = lambda: 0.0,
) -> Summary:
    """Run (or resume) an eval and write its summary.

    `make_executor(block_args_fn)` builds the executor; `persist()` is called
    after block files are written (e.g. volume commit); `block_seconds()`
    reports total worker time on blocks and `startup_seconds()` total
    container load/compile time, both for the summary.
    """
    lst = resolve_index_list(spec)
    run_dir = spec.run_dir(root)
    check_or_write_spec(run_dir, spec)
    completed = read_blocks(run_dir)
    stop = stop_from_dict(spec.stop)
    log(f"eval {run_dir}: {len(lst)} indices ({lst.name}), block_size={spec.block_size}, "
        f"workers={spec.n_workers}, max_in_flight={spec.window}, "
        f"{len(completed)} blocks already on disk")

    t0 = time.time()
    last_persist = [time.time()]

    def on_block_done(block: int, rows: list[Row]) -> None:
        write_block(run_dir, block, rows)
        if time.time() - last_persist[0] > 30:
            persist()
            last_persist[0] = time.time()

    def on_progress(counts: PrefixCounts, prefix_blocks: int) -> None:
        log(f"  prefix n={counts.n} correct={counts.correct} wrong={counts.wrong} "
            f"timeout={counts.timeout} ({prefix_blocks} blocks)")

    executor = make_executor(lambda b: block_args(spec, lst, b))
    result = run_blocks(
        n_items=len(lst), block_size=spec.block_size, executor=executor, stop=stop,
        max_in_flight=spec.window, completed=completed,
        on_block_done=on_block_done, on_progress=on_progress,
    )
    alpha = spec.stop.get("alpha", 0.05)
    summary = summarize(spec, run_dir, result, wall_seconds=time.time() - t0,
                        gpu_seconds=block_seconds(), startup_seconds=startup_seconds(),
                        alpha=alpha)
    write_summary(run_dir, summary)
    persist()
    return summary
