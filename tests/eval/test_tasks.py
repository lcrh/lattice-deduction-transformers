"""Each benchmark's eval task runs end to end on CPU (tiny checkpoint, synthetic data)."""


import numpy as np
import pytest

from lattice_diffusion.eval import indices as index_lists
from lattice_diffusion.eval.config import build_spec
from lattice_diffusion.eval.runner import block_args, solve_block
from lattice_diffusion.eval.tasks import get_task
from experiments.registry import experiment_class
from tests.equivalence import fakes, train_runs

ROW_KEYS = {"idx", "pos", "block", "status", "rounds", "calls", "forwards", "n_resets",
            "solution", "verified"}


def _small_solve(benchmark):
    exp = experiment_class(benchmark)()
    solve = exp.solve_dict()
    solve.update(max_rounds=3, n_chains=4, batch_size=8)
    if benchmark == "maze":
        solve["maze_data"].update(dataset="synthetic", grid_size=7, hard=False, n_puzzles=6)
    return exp, solve


@pytest.mark.parametrize("benchmark,split", [("sudoku", "test"), ("snowflake", "test"),
                                             ("maze", "synthetic")])
def test_task_solves_a_block(benchmark, split, tmp_path):
    tmp = tmp_path
    train_runs.RUNS[benchmark](tmp)
    ckpt = str(tmp / f"{benchmark}.pt")
    exp, solve = _small_solve(benchmark)
    task = get_task(benchmark, str(tmp))
    model = task.load_model(ckpt, 0.05, False)
    lst = index_lists.explicit(benchmark, split, np.arange(6)[::-1])
    spec = build_spec(benchmark, ckpt, exp.eval, solve,
                      explicit_indices=lst.indices.tolist(), explicit_split=split)
    spec.block_size = 4
    rows = solve_block(task, model, block_args(spec, lst, 1)).rows
    again = solve_block(task, model, block_args(spec, lst, 1)).rows
    assert [r["idx"] for r in rows] == [1, 0] and [r["pos"] for r in rows] == [4, 5]
    assert all(ROW_KEYS <= r.keys() for r in rows)
    assert all(r["status"] in {"correct", "wrong", "timeout"} for r in rows)
    assert rows == again        # block-seeded: rerunning a block reproduces it
    if benchmark == "maze":
        assert all("label" in r for r in rows)


@pytest.mark.parametrize("estimate", [False, True])
def test_drained_chain_estimate_is_recorded_only_when_enabled(estimate, tmp_path):
    tmp = tmp_path
    train_runs.RUNS["sudoku"](tmp)
    ckpt = str(tmp / "sudoku.pt")
    exp, solve = _small_solve("sudoku")
    solve["estimate_sequential"] = estimate
    task = get_task("sudoku", str(tmp))
    model = task.load_model(ckpt, 0.05, False)
    lst = index_lists.explicit("sudoku", "test", np.arange(4))
    spec = build_spec("sudoku", ckpt, exp.eval, solve,
                      explicit_indices=lst.indices.tolist(), explicit_split="test")
    rows = solve_block(task, model, block_args(spec, lst, 0)).rows
    assert all(("forwards_seq" in r) == estimate for r in rows)
