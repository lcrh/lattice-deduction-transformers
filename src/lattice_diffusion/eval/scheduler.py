"""Block scheduler: sequential stopping with bounded parallelism.

The index list is cut into fixed blocks. Blocks are dispatched in order to an
executor, at most `max_in_flight` beyond the contiguous completed prefix.
Statistics and the stopping rule only ever see that prefix, walked puzzle by
puzzle in index-list order, so the reported result does not depend on the
number of workers, the window size, or completion order. Those only change
wall-clock time and how much work runs past the stopping point.

The executor is abstract; Modal runs use a thread pool of blocking remote
calls (one GPU container per worker), tests use local functions.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Protocol

from lattice_diffusion.eval.stop import PrefixCounts, StopRule

Row = dict


class Executor(Protocol):
    def submit(self, block: int) -> None: ...

    def wait_any(self) -> list[tuple[int, list[Row]]]:
        """Block until at least one submitted block finishes; return all
        finished (block, rows) pairs not yet returned."""
        ...


@dataclass
class ScheduleResult:
    n: int                          # reported prefix length (puzzles)
    stop_reason: str                # "failures" | "half_width" | "fixed_n" | "exhausted"
    counts: PrefixCounts
    rows: list[Row]                 # the reported prefix, in index-list order
    n_evaluated: int                # every puzzle evaluated, including overshoot
    blocks_done: list[int] = field(default_factory=list)


def block_bounds(block: int, block_size: int, n_items: int) -> tuple[int, int]:
    start = block * block_size
    return start, min(start + block_size, n_items)


def n_blocks(n_items: int, block_size: int) -> int:
    return (n_items + block_size - 1) // block_size


def run_blocks(
    n_items: int,
    block_size: int,
    executor: Executor,
    stop: StopRule,
    max_in_flight: int,
    completed: dict[int, list[Row]] | None = None,
    on_block_done: Callable[[int, list[Row]], None] | None = None,
    on_progress: Callable[[PrefixCounts, int], None] | None = None,
) -> ScheduleResult:
    """Evaluate blocks until the stop rule fires or the list is exhausted.

    `completed` holds blocks already on disk (resume); they are not
    resubmitted. `on_block_done` is called for every newly finished block
    (persist it); `on_progress` whenever the contiguous prefix advances.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    total_blocks = n_blocks(n_items, block_size)
    # A rule with a fixed `limit` never needs blocks past the one holding its
    # last item. Blocks stay full-size, so they are shared with other rules.
    if stop.limit is not None:
        total_blocks = min(total_blocks, n_blocks(stop.limit, block_size))
    done: dict[int, list[Row]] = dict(completed or {})
    in_flight: set[int] = set()
    next_block = 0          # lowest block id not yet submitted or done
    prefix_blocks = 0       # blocks 0..prefix_blocks-1 are all done
    counts = PrefixCounts()
    prefix_rows: list[Row] = []
    stop_reason: str | None = None

    def advance_prefix() -> None:
        nonlocal prefix_blocks, stop_reason
        while stop_reason is None and prefix_blocks in done:
            rows = done[prefix_blocks]
            expected = block_bounds(prefix_blocks, block_size, n_items)
            if len(rows) != expected[1] - expected[0]:
                raise RuntimeError(
                    f"block {prefix_blocks} has {len(rows)} rows, "
                    f"expected {expected[1] - expected[0]}")
            for row in rows:
                counts.add(row["status"])
                prefix_rows.append(row)
                reason = stop.should_stop(counts)
                if reason is not None:
                    stop_reason = reason
                    break
            prefix_blocks += 1
        if on_progress is not None:
            on_progress(counts, prefix_blocks)

    def fill_window() -> None:
        nonlocal next_block
        while (stop_reason is None and next_block < total_blocks
               and next_block < prefix_blocks + max_in_flight):
            if next_block not in done:
                executor.submit(next_block)
                in_flight.add(next_block)
            next_block += 1

    advance_prefix()
    fill_window()
    while in_flight:
        for block, rows in executor.wait_any():
            in_flight.discard(block)
            done[block] = rows
            if on_block_done is not None:
                on_block_done(block, rows)
        advance_prefix()
        fill_window()

    if stop_reason is None:
        if prefix_blocks < total_blocks:
            raise RuntimeError("scheduler finished without covering all blocks")
        stop_reason = "exhausted"
    return ScheduleResult(
        n=counts.n,
        stop_reason=stop_reason,
        counts=counts,
        rows=prefix_rows,
        n_evaluated=sum(len(r) for r in done.values()),
        blocks_done=sorted(done),
    )


class ThreadExecutor:
    """Runs `solve_block(block) -> rows` on a local thread pool."""

    def __init__(self, solve_block: Callable[[int], list[Row]], n_workers: int):
        self._solve_block = solve_block
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        self._futures: dict[Future, int] = {}

    def submit(self, block: int) -> None:
        self._futures[self._pool.submit(self._solve_block, block)] = block

    def wait_any(self) -> list[tuple[int, list[Row]]]:
        finished, _ = wait(list(self._futures), return_when=FIRST_COMPLETED)
        out = []
        for fut in finished:
            block = self._futures.pop(fut)
            out.append((block, fut.result()))
        return out

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
