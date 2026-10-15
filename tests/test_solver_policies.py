"""Behavior of the solver options (deduction passes, decision and
backtracking policies). Their defaults are covered bit for bit by
tests/equivalence."""

import pytest
import torch

from experiments.sudoku.dpll import StepConfig, _alive_entropy, dpll_step
from experiments.sudoku.solve import DecisionStack, SolveConfig, solve
from tests.equivalence import fakes

DPLL = torch.load(fakes.FIXTURES / "dpll.pt", weights_only=False)


def _case(name="sudoku/aug"):
    ref = DPLL[name]
    inp = ref["inputs"]
    model = fakes.tiny_model(seed=1, **inp["model_kw"])
    model.load_state_dict(ref["model_state"])
    model.eval()
    return model, inp


def _step(model, inp, seed=0, decide_rank=None, **kw):
    torch.manual_seed(seed)
    return dpll_step(model, inp["state"], inp["given"],
                     StepConfig(**{**inp["step"], "augment": False, **kw}),
                     in_puzzle_mask=inp["ip"], decide_rank=decide_rank)


def _decisions(inp, new_state, info):
    """(row, cell, digit) of each decision taken in the step."""
    post = inp["state"].masked_fill(info["deduce_mask"], 0.0)
    changed = (post.sum(-1) > 1) & (new_state.sum(-1) == 1)
    return [(b, int(c), int(new_state[b, c].argmax())) for b, c in changed.nonzero().tolist()]


def test_bad_policy_names_are_rejected():
    with pytest.raises(ValueError):
        StepConfig(cell_policy="random")
    with pytest.raises(ValueError):
        StepConfig(deduce_passes=0)
    with pytest.raises(ValueError):
        SolveConfig(backtrack="undo")
    with pytest.raises(ValueError):
        SolveConfig(backtrack_negate=True)


def test_argmax_digit_policy_matches_zero_temperature_sampling():
    model, inp = _case()
    a = _step(model, inp, digit_policy="argmax")
    b = _step(model, inp, digit_policy="sample", temp_decide=0.0)
    assert torch.equal(a[0], b[0])


def test_rank_k_takes_the_kth_best_candidate():
    model, inp = _case()
    B = inp["state"].shape[0]
    zero = _step(model, inp, seed=3, digit_policy="rank_k", decide_rank=torch.zeros(B, dtype=torch.long))
    argmax = _step(model, inp, seed=3, digit_policy="argmax")
    assert torch.equal(zero[0], argmax[0])
    new_state, _, _, out, info = _step(model, inp, seed=3, digit_policy="rank_k",
                                       decide_rank=torch.ones(B, dtype=torch.long))
    post = inp["state"].masked_fill(info["deduce_mask"], 0.0)
    for b, cell, digit in _decisions(inp, new_state, info):
        alive = post[b, cell] > 0.5
        logits = out["softmax"][b, cell].masked_fill(~alive, float("-inf"))
        assert digit == int(logits.argsort(descending=True)[min(1, int(alive.sum()) - 1)])


@pytest.mark.parametrize("policy", ["mrv", "min_entropy", "max_entropy"])
def test_scored_cell_policies_pick_an_extreme_cell(policy):
    model, inp = _case()
    new_state, _, _, out, info = _step(model, inp, cell_policy=policy)
    post = inp["state"].masked_fill(info["deduce_mask"], 0.0)
    decisions = _decisions(inp, new_state, info)
    assert decisions
    for b, cell, _ in decisions:
        multi = post[b].sum(-1) > 1
        if policy == "mrv":
            score = post[b].sum(-1)
        else:
            score = _alive_entropy(post[b:b + 1], out["softmax"][b:b + 1], 9)[0]
        pick = score[cell]
        best = score[multi].min() if policy != "max_entropy" else score[multi].max()
        assert torch.isclose(pick, best, atol=1e-5)


@pytest.mark.parametrize("passes", [2, 4, "fixpoint"])
def test_multi_pass_deduction(passes):
    model, inp = _case()
    new_state, conflict, solved, _, info = _step(model, inp, deduce_passes=passes)
    masks = info["pass_deduce_masks"]
    assert info["n_passes"] == len(masks) <= (16 if passes == "fixpoint" else passes)
    union = torch.zeros_like(masks[0])
    for m in masks:
        assert not (m & union).any()          # each pass eliminates new candidates
        union |= m
    assert torch.equal(info["deduce_mask"], union)
    if passes == "fixpoint" and info["n_passes"] < 16:
        assert not masks[-1].any() or bool((conflict | solved).all())


def test_eval_loop_override_matches_a_model_built_with_that_depth():
    model, inp = _case()
    shallow = fakes.tiny_model(seed=1, n_loops=1)
    shallow.load_state_dict(model.state_dict())
    shallow.eval()
    a = _step(model, inp, seed=4, n_loops=1)
    b = _step(shallow, inp, seed=4)
    assert torch.equal(a[3]["bce"], b[3]["bce"]) and torch.equal(a[0], b[0])


def _hard_solve(**solve_kw):
    x, y = fakes.sudoku_tensors(6, (25, 40), seed=41)
    x, y = x.float(), y.float()
    model = fakes.tiny_model(seed=3)
    model.train()
    cfg = SolveConfig(step=StepConfig(cls_threshold=0.6, augment=True), max_rounds=25,
                      n_chains=4, batch_size=8, **solve_kw)
    torch.manual_seed(9)
    return solve(model, x, y, x.sum(-1) == 1, cfg)


@pytest.mark.parametrize("backtrack,negate", [("last", False), ("geometric", False),
                                              ("uniform_depth", False), ("last", True)])
def test_backtracking_policies(backtrack, negate):
    res = _hard_solve(backtrack=backtrack, backtrack_negate=negate, max_decision_depth=8)
    assert torch.equal(res.correct | res.wrong | res.timeouts, torch.ones_like(res.correct))
    assert res.search.backtracks == int(res.n_resets.sum())
    assert res.search.sum_target_depth <= res.search.sum_conflict_depth


@pytest.mark.parametrize("negate,gt_digit", [(False, 2), (True, 2), (True, 1)])
def test_decision_stack_backtracks_to_the_last_decision(negate, gt_digit):
    state = torch.tensor([[[1., 1., 1.], [1., 1., 0.]]])          # cell 0: {0,1,2}; cell 1: {0,1}
    new_state = torch.tensor([[[0., 1., 0.], [1., 1., 0.]]])      # decided cell 0 = 1
    original = torch.ones(1, 2, 3)
    stack = DecisionStack(1, 2, 3, max_depth=4, device="cpu")
    stack.push(state, torch.zeros_like(state, dtype=torch.bool), new_state,
               torch.tensor([True]), vd=3)
    assert int(stack.depth[0]) == 1
    cfg = SolveConfig(backtrack="last", backtrack_negate=negate)
    out = new_state.clone()
    at_root = stack.backtrack(torch.tensor([0]), cfg, out, original,
                              torch.tensor([gt_digit, 0]), vd=3)
    expected = state.clone()
    if negate:
        expected[0, 0, 1] = 0.0                # the decided candidate is ruled out
    assert torch.equal(out, expected)
    assert bool(at_root[0]) == (not negate)    # back at depth 0, but negation is progress
    assert stack.stats.negations == int(negate)
    assert stack.stats.unsound_negations == int(negate and gt_digit == 1)


def test_per_puzzle_diagnostics_add_up_to_the_totals():
    res = _hard_solve()
    totals = res.puzzle_diag.sum(dim=0).tolist()
    assert totals == [res.diag_total_deduced, res.diag_total_unsound_deductions,
                      res.diag_conflict_tp, res.diag_conflict_fp, res.diag_conflict_fn,
                      res.diag_conflict_tn]


def test_multi_pass_solve_reports_per_pass_diagnostics():
    x, y = fakes.sudoku_tensors(6, (25, 40), seed=41)
    x, y = x.float(), y.float()
    model = fakes.tiny_model(seed=3)
    cfg = SolveConfig(step=StepConfig(cls_threshold=0.6, augment=True, deduce_passes=2),
                      max_rounds=25, n_chains=4, batch_size=8)
    torch.manual_seed(9)
    res = solve(model, x, y, x.sum(-1) == 1, cfg)
    assert res.puzzle_pass_diag.shape == (6, 2, 2)
    assert torch.equal(res.puzzle_pass_diag.sum(dim=1), res.puzzle_diag[:, :2])
