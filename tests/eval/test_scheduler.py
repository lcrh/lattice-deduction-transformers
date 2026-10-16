import random

import pytest

from lattice_diffusion.eval.scheduler import ThreadExecutor, n_blocks, run_blocks
from lattice_diffusion.eval.stop import PRESETS, FixedN, NoStop, SequentialStop, stop_from_dict

from tests.eval.fakes import ReorderExecutor, block_solver, make_spec


def _run(spec, executor, max_in_flight, completed=None, on_progress=None):
    return run_blocks(
        n_items=len(spec.explicit_indices), block_size=spec.block_size, executor=executor,
        stop=stop_from_dict(spec.stop), max_in_flight=max_in_flight, completed=completed,
        on_progress=on_progress,
    )


@pytest.fixture(scope="module")
def reference():
    spec = make_spec()
    return spec, _run(spec, ReorderExecutor(block_solver(spec)), max_in_flight=1)


def test_reference_stops_early_on_failures(reference):
    spec, ref = reference
    assert ref.stop_reason == "failures"
    assert ref.counts.failures == 30
    assert 200 <= ref.n < len(spec.explicit_indices)
    assert [r["pos"] for r in ref.rows] == list(range(ref.n))


@pytest.mark.parametrize("workers,window", [(1, 1), (4, 1), (4, 4), (16, 16), (16, 64)])
def test_result_independent_of_parallelism(reference, workers, window):
    spec, ref = reference
    executor = ThreadExecutor(block_solver(spec, jitter_s=0.002), n_workers=workers)
    try:
        res = _run(spec, executor, max_in_flight=window)
    finally:
        executor.shutdown()
    assert (res.n, res.stop_reason, res.rows) == (ref.n, ref.stop_reason, ref.rows)
    assert res.n_evaluated - res.n < window * spec.block_size + spec.block_size


@pytest.mark.parametrize("seed", range(5))
def test_result_independent_of_completion_order(reference, seed):
    spec, ref = reference
    res = _run(spec, ReorderExecutor(block_solver(spec), seed=seed), max_in_flight=8)
    assert (res.n, res.stop_reason, res.rows) == (ref.n, ref.stop_reason, ref.rows)


@pytest.mark.parametrize("seed", range(5))
def test_resume_from_any_completed_subset(reference, seed):
    """Resuming from any set of finished blocks gives the same result and never redoes them."""
    spec, ref = reference
    solve = block_solver(spec)
    rng = random.Random(seed)
    total = n_blocks(len(spec.explicit_indices), spec.block_size)
    on_disk = {b: solve(b) for b in rng.sample(range(total), k=rng.randint(0, 25))}
    executor = ReorderExecutor(solve, seed=seed)
    res = _run(spec, executor, max_in_flight=4, completed=on_disk)
    assert (res.n, res.stop_reason, res.rows) == (ref.n, ref.stop_reason, ref.rows)
    assert not set(executor.submitted) & set(on_disk)


@pytest.mark.parametrize("window", [1, 3, 8])
def test_slow_block_bounds_overshoot(window):
    """A straggler holds the window, so work past the stopping point is at most `window` blocks."""
    spec = make_spec()
    executor = ReorderExecutor(block_solver(spec), seed=0, slow=frozenset({2, 5}))
    prefix = [0]
    real_submit = executor.submit

    def checked_submit(block):
        assert block < prefix[0] + window
        real_submit(block)

    executor.submit = checked_submit
    res = _run(spec, executor, max_in_flight=window,
               on_progress=lambda counts, blocks: prefix.__setitem__(0, blocks))
    assert res.n_evaluated - res.n < (window + 1) * spec.block_size


def test_no_stop_evaluates_everything():
    spec = make_spec(stop=NoStop())
    res = _run(spec, ReorderExecutor(block_solver(spec), seed=3), max_in_flight=6)
    assert res.stop_reason == "exhausted"
    assert res.n == res.n_evaluated == len(spec.explicit_indices)
    assert [r["pos"] for r in res.rows] == list(range(res.n))


@pytest.mark.parametrize("workers,window", [(1, 1), (4, 1), (4, 4), (16, 16), (16, 64)])
def test_fixed_n_reports_exactly_n(workers, window):
    """Dispatch stops at the block holding item n, whatever the parallelism."""
    spec = make_spec(stop=FixedN(1000))
    executor = ThreadExecutor(block_solver(spec, jitter_s=0.002), n_workers=workers)
    try:
        res = _run(spec, executor, max_in_flight=window)
    finally:
        executor.shutdown()
    assert res.stop_reason == "fixed_n"
    assert res.n == len(res.rows) == 1000
    assert [r["pos"] for r in res.rows] == list(range(1000))
    assert res.n_evaluated == n_blocks(1000, spec.block_size) * spec.block_size


def test_fixed_n_longer_than_list_is_exhausted():
    spec = make_spec(stop=FixedN(10_000))
    res = _run(spec, ReorderExecutor(block_solver(spec), seed=1), max_in_flight=4)
    assert res.stop_reason == "exhausted"
    assert res.n == len(spec.explicit_indices)


def test_fixed_n_round_trips_and_is_the_ablation_preset():
    assert stop_from_dict(FixedN(1000).to_dict()) == FixedN(1000)
    assert PRESETS["ablation"] == SequentialStop(n_max=4096)


def test_failed_block_drains_in_flight_blocks_then_raises():
    spec = make_spec(stop=NoStop())
    solve = block_solver(spec)

    def flaky(block):
        if block == 3:
            raise ValueError("boom")
        return solve(block)

    persisted = []
    executor = ThreadExecutor(flaky, n_workers=4)
    with pytest.raises(RuntimeError, match="block 3 failed"):
        run_blocks(n_items=len(spec.explicit_indices), block_size=spec.block_size,
                   executor=executor, stop=NoStop(), max_in_flight=8,
                   on_block_done=lambda b, rows: persisted.append(b))
    executor.shutdown()
    assert 3 not in persisted
    assert set(range(8)) - {3} <= set(persisted)    # the rest of the window was kept


def test_sequential_stop_caps_at_n_max():
    from lattice_diffusion.eval.stop import PRESETS, PrefixCounts, SequentialStop, stop_from_dict

    rule = SequentialStop(n_max=4000)
    assert rule.limit == 4000 and stop_from_dict(rule.to_dict()) == rule
    counts = PrefixCounts(n=3999, correct=3000, timeout=999)  # many failures, but past n_min
    assert rule.should_stop(counts) == "failures"
    assert rule.should_stop(PrefixCounts(n=4000, correct=4000)) == "n_max"
    assert PRESETS["select"] == rule


def test_failure_rule_fires_before_n_min():
    from lattice_diffusion.eval.stop import PrefixCounts, SequentialStop

    rule = SequentialStop()
    assert rule.should_stop(PrefixCounts(n=300, correct=200, timeout=100)) == "failures"
    assert rule.should_stop(PrefixCounts(n=500, correct=500)) is None   # half-width waits for n_min
