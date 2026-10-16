import dataclasses
import json

import pytest

from lattice_diffusion.eval.records import (
    check_or_write_spec, format_summary, read_blocks, summary_path,
)
from lattice_diffusion.eval.runner import run_eval
from lattice_diffusion.eval.scheduler import ThreadExecutor
from lattice_diffusion.eval.stop import NoStop

from tests.eval.fakes import block_solver, make_spec


def test_spec_hash_ignores_cost_only_fields():
    spec = make_spec()
    same = dataclasses.replace(spec, n_workers=100, max_in_flight=7, keep_warm_s=10,
                               stop=NoStop().to_dict())
    assert same.spec_hash() == spec.spec_hash()
    assert dataclasses.replace(spec, eval_seed=1).spec_hash() != spec.spec_hash()
    assert dataclasses.replace(spec, block_size=128).spec_hash() != spec.spec_hash()


def test_run_dir_refuses_a_different_spec(tmp_path):
    spec = make_spec()
    run_dir = tmp_path / "run"
    check_or_write_spec(run_dir, spec)
    check_or_write_spec(run_dir, dataclasses.replace(spec, n_workers=3))
    with pytest.raises(RuntimeError):
        check_or_write_spec(run_dir, dataclasses.replace(spec, eval_seed=5))


def _threads(spec):
    return lambda block_args: ThreadExecutor(block_solver(spec), n_workers=4)


def test_run_eval_writes_summary_and_resumes(tmp_path):
    spec = make_spec(n_workers=4)
    summary = run_eval(spec, tmp_path, _threads(spec), log=lambda _: None)
    run_dir = spec.run_dir(tmp_path)
    on_disk = json.loads(summary_path(run_dir, spec.stop).read_text())
    assert on_disk["n"] == summary.n and on_disk["stop_reason"] == "failures"
    assert summary.correct + summary.wrong + summary.timeout == summary.n
    assert summary.accuracy_cs[0] < summary.accuracy < summary.accuracy_cs[1]
    blocks = read_blocks(run_dir)
    assert sum(len(r) for r in blocks.values()) == summary.n_evaluated

    submitted = []

    def recording(spec):
        class Recording(ThreadExecutor):
            def submit(self, block):
                submitted.append(block)
                super().submit(block)
        return lambda _: Recording(block_solver(spec), n_workers=4)

    again = run_eval(spec, tmp_path, recording(spec), log=lambda _: None)
    assert submitted == []
    assert (again.n, again.correct, again.wrong) == (summary.n, summary.correct, summary.wrong)

    # A full eval of the same spec reuses the sequential run's blocks.
    full = dataclasses.replace(spec, stop=NoStop().to_dict())
    assert full.run_dir(tmp_path) == run_dir
    result = run_eval(full, tmp_path, recording(full), log=lambda _: None)
    assert set(submitted).isdisjoint(blocks)
    assert result.n == len(spec.explicit_indices) and result.stop_reason == "exhausted"
    assert summary_path(run_dir, full.stop).exists()



def test_block_stats_are_saved_per_block_and_summed(tmp_path):
    spec = make_spec(n_workers=4)
    summary = run_eval(spec, tmp_path, _threads(spec), log=lambda _: None,
                       block_meta=lambda block: {"model_calls": 10, "seconds": 2.0,
                                                  "startup_seconds": 1.0})
    blocks = read_blocks(spec.run_dir(tmp_path))
    n_blocks = len(blocks)
    assert summary.model_calls_total == 10 * n_blocks
    assert summary.gpu_seconds == 2.0 * n_blocks
    assert summary.startup_seconds == 1.0 * n_blocks
    # Calls per solve covers only the blocks the reported prefix touches.
    reported = range(summary.n // spec.block_size + (summary.n % spec.block_size > 0))
    correct = sum(r["status"] == "correct" for b in reported for r in blocks[b])
    assert summary.calls_per_solve == pytest.approx(10 * len(reported) / correct)
    assert summary.mean_resets == 0 and summary.mean_rounds_correct == 1
    assert "model calls=" in format_summary(summary)


def test_inverse_binomial_rate_only_for_runs_stopped_on_failures(tmp_path):
    spec = make_spec()
    seq = run_eval(spec, tmp_path, _threads(spec), log=lambda _: None)
    assert seq.stop_reason == "failures"
    failures = seq.wrong + seq.timeout
    assert seq.failure_rate_inverse_binomial == pytest.approx((failures - 1) / (seq.n - 1))
    full = run_eval(dataclasses.replace(spec, stop=NoStop().to_dict()), tmp_path, _threads(spec),
                    log=lambda _: None)
    assert full.failure_rate_inverse_binomial == full.failure_rate
