"""The refactored code reproduces the pre-refactor outputs bit for bit.

Fixtures were captured on CPU from the code at commit 28dbc3b (before the
config refactor) with seeded tiny models and synthetic data; see
tests/equivalence/README.md.
"""


import pytest
import torch

from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.solve import SolveConfig, solve
from tests.equivalence import fakes
from tests.equivalence import train_runs


def _load(name):
    return torch.load(fakes.FIXTURES / f"{name}.pt", weights_only=False)


DPLL = _load("dpll")
SOLVE = _load("solve")


@pytest.mark.parametrize("case", sorted(DPLL))
def test_dpll_step_matches_reference(case):
    ref = DPLL[case]
    inp = ref["inputs"]
    model = fakes.tiny_model(seed=1, **inp["model_kw"])
    model.load_state_dict(ref["model_state"])
    model.train()
    torch.manual_seed(123)
    new_state, conflict, solved, out, info = dpll_step(
        model, inp["state"], inp["given"], StepConfig(**inp["step"]),
        in_puzzle_mask=inp["ip"])
    assert torch.equal(new_state, ref["new_state"])
    assert torch.equal(conflict, ref["conflict"])
    assert torch.equal(solved, ref["solved"])
    assert torch.equal(info["deduce_mask"], ref["deduce_mask"])
    assert torch.equal(out["bce"], ref["bce"])
    assert torch.equal(out["softmax"], ref["softmax"])
    assert torch.equal(out["conflict"], ref["conflict_logit"])


@pytest.mark.parametrize("case", sorted(SOLVE))
def test_solve_matches_reference(case):
    ref = SOLVE[case]
    inp = ref["inputs"]
    model = fakes.tiny_model(seed=2)
    model.load_state_dict(ref["model_state"])
    model.train()
    cfg = SolveConfig(step=StepConfig(**inp["step"]), **inp["solve"])
    torch.manual_seed(321)
    res = solve(model, inp["x"], inp["y"], inp["given"], cfg)
    for key in ("correct", "wrong", "timeouts", "round_solved", "n_resets", "solution",
                "puzzle_calls"):
        assert torch.equal(getattr(res, key), ref[key]), key
    assert res.model_calls == ref["model_calls"]
    diag = (res.diag_total_deduced, res.diag_total_unsound_deductions,
            res.diag_conflict_tp, res.diag_conflict_fp, res.diag_conflict_fn,
            res.diag_conflict_tn, res.diag_active_chain_rounds)
    assert diag == ref["diag"]



def test_verbose_solve_logs_each_puzzle_and_matches_reference(capsys):
    case = sorted(SOLVE)[0]
    ref = SOLVE[case]
    inp = ref["inputs"]
    model = fakes.tiny_model(seed=2)
    model.load_state_dict(ref["model_state"])
    model.train()
    cfg = SolveConfig(step=StepConfig(**inp["step"]), **inp["solve"])
    torch.manual_seed(321)
    res = solve(model, inp["x"], inp["y"], inp["given"], cfg, verbose=True)
    assert torch.equal(res.solution, ref["solution"]) and res.model_calls == ref["model_calls"]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 + inp["x"].shape[0]      # header + one per puzzle
    print("\n".join(lines[:3]))

@pytest.mark.parametrize("benchmark", ["sudoku", "snowflake", "maze"])
def test_training_matches_reference(benchmark, tmp_path):
    ref = _load("train")[benchmark]
    state = train_runs.RUNS[benchmark](tmp_path)
    assert fakes.state_dict_equal(state, ref)
