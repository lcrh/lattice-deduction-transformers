"""Backend-independent eval driver.

`run_eval` resolves the index list, resumes from blocks on disk, schedules the
rest through an executor and writes the summary; `solve_block` is what a worker
runs for one block. Used by the Modal app and by local runs and tests.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import torch

from lattice_diffusion.eval import indices as index_lists
from lattice_diffusion.eval.records import (
    EvalSpec, Summary, check_or_write_digest, check_or_write_spec, read_blocks, summarize,
    write_block, write_summary,
)
from lattice_diffusion.eval.scheduler import Executor, Row, block_bounds, run_blocks
from lattice_diffusion.eval.stop import PrefixCounts, stop_from_dict
from lattice_diffusion.eval.tasks import EvalTask, TaskBlock

PERSIST_EVERY_S = 30    # seconds between volume commits while blocks finish

SplitSize = Callable[[str], int]
FilterIndices = Callable[[str, np.ndarray, dict], np.ndarray]


class BlockArgs(TypedDict):
    """Everything a worker needs for one block (plain types, sent over RPC)."""

    solve: dict
    eval_seed: int
    block: int
    start: int
    split: str
    split_size: int | None
    indices: list[int]


def resolve_index_list(spec: EvalSpec, split_size: SplitSize | None = None,
                       filter_indices: FilterIndices | None = None) -> index_lists.IndexList:
    """The index list `spec` walks, after its index_filter.

    Args:
        split_size: Sizes unpinned named lists and checks pinned ones.
        filter_indices: Applies spec.index_filter.
    """
    lst = _named_or_explicit(spec, split_size)
    if spec.index_filter is None:
        return lst
    if filter_indices is None:
        raise ValueError("spec has an index_filter but no filter_indices was given")
    kept = np.asarray(filter_indices(lst.split, lst.indices, spec.index_filter), dtype=np.int64)
    tag = index_lists.filter_tag(spec.index_filter)
    return index_lists.IndexList(name=f"{lst.name}[{tag}]", task=lst.task, split=lst.split,
                                 indices=kept, split_size=lst.split_size)


def _named_or_explicit(spec: EvalSpec, split_size: SplitSize | None) -> index_lists.IndexList:
    if spec.index_list == "explicit":
        if spec.explicit_indices is None or spec.explicit_split is None:
            raise ValueError("explicit index list needs explicit_indices and explicit_split")
        return index_lists.explicit(spec.task, spec.explicit_split, spec.explicit_indices)
    list_spec = index_lists.SPECS[spec.index_list]
    n = split_size(list_spec.split) if split_size else None
    lst = index_lists.load(spec.index_list, split_size=n)
    if lst.task != spec.task:
        raise ValueError(f"index list {spec.index_list} is for task {lst.task}, not {spec.task}")
    return lst


def block_seed(eval_seed: int, block: int) -> int:
    """63-bit seed for one block, derived only from (eval_seed, block)."""
    digest = hashlib.sha256(f"{eval_seed}:{block}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def block_args(spec: EvalSpec, lst: index_lists.IndexList, block: int) -> BlockArgs:
    start, end = block_bounds(block, spec.block_size, len(lst))
    return BlockArgs(solve=spec.solve, eval_seed=spec.eval_seed, block=block, start=start,
                     split=lst.split, split_size=lst.split_size,
                     indices=lst.indices[start:end].tolist())


def solve_block(task: EvalTask, model, args: BlockArgs) -> TaskBlock:
    """Seed from (eval_seed, block), solve, and tag rows with pos/block."""
    if args["split_size"] is not None:
        actual = task.split_size(args["split"])
        if actual != args["split_size"]:
            raise ValueError(f"{args['split']} split has {actual} items, "
                             f"index list expects {args['split_size']}")
    seed = block_seed(args["eval_seed"], args["block"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    out = task.solve_block(model, args["split"], np.asarray(args["indices"], dtype=np.int64),
                           args["solve"])
    for k, row in enumerate(out.rows):
        row["pos"] = args["start"] + k
        row["block"] = args["block"]
    return out


def run_eval(
    spec: EvalSpec,
    root: str | Path,
    make_executor: Callable[[Callable[[int], BlockArgs]], Executor],
    persist: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
    block_meta: Callable[[int], dict | None] = lambda block: None,
    split_size: SplitSize | None = None,
    filter_indices: FilterIndices | None = None,
) -> Summary:
    """Run (or resume) an eval and write its summary.

    Args:
        make_executor: Builds the executor from a block -> BlockArgs function.
        persist: Called after block files are written (e.g. a volume commit).
        block_meta: Per-block worker stats ({"model_calls", "seconds", "startup_seconds"})
            saved next to the block, so the summary's totals survive resumes.
        split_size, filter_indices: See resolve_index_list.
    """
    lst = resolve_index_list(spec, split_size, filter_indices)
    run_dir = spec.run_dir(root)
    check_or_write_spec(run_dir, spec)
    check_or_write_digest(run_dir, lst.digest, lst.name)
    if lst.name in index_lists.SPECS and not index_lists.SPECS[lst.name].pinned:
        log(f"  note: {lst.name} is unpinned ({len(lst)} items, sha256 {lst.digest}); "
            f"pin it in lattice_diffusion/eval/indices.py")
    completed = read_blocks(run_dir)
    stop = stop_from_dict(spec.stop)
    log(f"eval {run_dir}: {len(lst)} indices ({lst.name}), block_size={spec.block_size}, "
        f"workers={spec.n_workers}, max_in_flight={spec.window}, "
        f"{len(completed)} blocks already on disk")

    t0 = last_persist = time.time()

    def on_block_done(block: int, rows: list[Row]) -> None:
        nonlocal last_persist
        write_block(run_dir, block, rows, block_meta(block))
        if time.time() - last_persist > PERSIST_EVERY_S:
            persist()
            last_persist = time.time()

    def on_progress(counts: PrefixCounts, prefix_blocks: int) -> None:
        log(f"  prefix n={counts.n} correct={counts.correct} wrong={counts.wrong} "
            f"timeout={counts.timeout} ({prefix_blocks} blocks)")

    executor = make_executor(lambda b: block_args(spec, lst, b))
    try:
        result = run_blocks(
            n_items=len(lst), block_size=spec.block_size, executor=executor, stop=stop,
            max_in_flight=spec.window, completed=completed,
            on_block_done=on_block_done, on_progress=on_progress,
        )
    finally:
        executor.shutdown()
        persist()
    summary = summarize(spec, run_dir, result, wall_seconds=time.time() - t0,
                        alpha=getattr(stop, "alpha", 0.05))
    write_summary(run_dir, summary)
    persist()
    return summary
