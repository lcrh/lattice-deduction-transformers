"""Tables (paired deltas), portfolio math and compute formulas on synthetic rows."""

import json
import random

import pytest

from paper import portfolio as P
from paper.compute import ldt_forward_flops
from paper.tables import appendix
from paper.tables.common import eval_rule, paired_vs


def _write_rows(tmp_path, key, statuses):
    path = tmp_path / f"{key}.jsonl"
    path.write_text("".join(json.dumps({"idx": 1000 + p, "pos": p, "status": s, "calls": p + 1, "forwards_seq": 64 * (p + 1),
                                        "solution": f"s{p}"}) + "\n"
                            for p, s in enumerate(statuses)))
    return path.name


def _record(tmp_path, entry, seed, statuses, **kw):
    n = len(statuses)
    correct = statuses.count("correct")
    return {"entry": entry, "key": f"A1/{entry}/seed{seed}", "seed": seed, "status": "done",
            "label": kw.pop("label", entry), "group": kw.pop("group", ""), "n": n,
            "accuracy": correct / n, "wrong": statuses.count("wrong"),
            "timeout": statuses.count("timeout"), "calls_mean": 3.0, "unsound_rate": 0.001,
            "stop_reason": "fixed_n",
            "rows_path": _write_rows(tmp_path, f"{entry}_{seed}", statuses), **kw}


def test_paired_delta_pools_seeds_by_puzzle(tmp_path):
    ref = [_record(tmp_path, "default", s, ["correct"] * 10) for s in (0, 1)]
    # Seed 0 fails puzzles 0-1, seed 1 fails puzzle 0 -> 3 of 20 pairs differ.
    arm = [_record(tmp_path, "arm", 0, ["timeout"] * 2 + ["correct"] * 8),
           _record(tmp_path, "arm", 1, ["wrong"] + ["correct"] * 9)]
    d = paired_vs(ref, arm, root=tmp_path)
    assert d.n == 20 and d.only_b == 3 and d.only_a == 0
    assert d.delta == pytest.approx(-0.15)
    assert d.lo < d.delta < d.hi


def test_appendix_table_has_reference_groups_and_deltas(tmp_path):
    records = [_record(tmp_path, "default", s, ["correct"] * 10, label="Default") for s in (0, 1)]
    records += [_record(tmp_path, "loops_1", s, ["timeout"] * 5 + ["correct"] * 5,
                        label="1 loop", group="loops") for s in (0, 1)]
    hints = {"reference": "default", "groups": {"loops": "Loop count"}}
    tex = appendix.build("appendix_a_recurrence", records, hints, root=tmp_path).to_latex()
    assert "Default & 2 & 100.0 $\\pm$ 0.0 & (ref.)" in tex
    assert "\\emph{Loop count}" in tex
    assert "1 loop & 2 & 50.0 $\\pm$ 0.0 & -50.0 [" in tex
    assert "first 10 puzzles" in tex


def test_eval_rule_caption():
    assert "until 100 failures (n = 1,000--5,000" in eval_rule(
        [{"stop_reason": "failures", "n": 1000}, {"stop_reason": "failures", "n": 5000}])


def _attempt(model, statuses, eval_seed=0, full=False, idx=None, solutions=None):
    idx = idx or list(range(len(statuses)))
    solutions = solutions or [f"ok{i}" if s == "correct" else (f"bad{i}{model}" if s == "wrong"
                              else "") for i, s in zip(idx, statuses)]
    rows = {i: {"idx": i, "status": s, "solution": sol}
            for i, s, sol in zip(idx, statuses, solutions)}
    return P.Attempt(model=model, eval_seed=eval_seed, full=full, rows=rows)


def test_portfolio_residue_and_selection():
    # 10 puzzles; anchor 0 fails {2, 5, 7}; anchor 1 fails {5}.
    a0 = _attempt(0, ["correct"] * 2 + ["timeout"] + ["correct"] * 2 + ["timeout", "correct",
                                                                         "wrong", "correct",
                                                                         "correct"], full=True)
    a1 = _attempt(1, ["correct"] * 5 + ["timeout"] + ["correct"] * 4, full=True)
    h = P.failure_set([a0, a1])
    assert h == [2, 5, 7]
    # Non-anchor model 2 is only run on H: solves 5, wrong on 2, times out on 7.
    m2 = _attempt(2, ["wrong", "correct", "timeout"], idx=h)
    assert P.residue_curve([a0, a1, m2]) == [3, 1, 0]
    assert P.residue_curve([a1, m2]) == [1, 0]
    with pytest.raises(ValueError):
        P.residue_curve([m2, a0])
    first = P.select([a0, a1, m2], h, "first")
    # 2: a0 timeout, a1 correct -> correct; 5: a0, a1 timeout, m2 correct;
    # 7: a0 wrong first -> wrong.
    assert first == {"correct": 2, "wrong": 1}
    major = P.select([a0, a1, m2], h, "majority")
    # 2: a1 correct vs m2 wrong -> tie -> abstain; 7: a0 wrong vs a1 correct -> tie.
    assert major == {"unsolved": 2, "correct": 1}
    probs = P.solve_probability([a0, a1, m2], h)
    assert probs[5] == (1, 3)


def test_portfolio_attempts_curve_and_analyze():
    full = ["correct"] * 8 + ["timeout", "timeout"]
    a0 = _attempt(0, full, full=True)
    a0_e1 = _attempt(0, ["correct", "timeout"], eval_seed=1, idx=[8, 9])
    a0_e2 = _attempt(0, ["timeout", "correct"], eval_seed=2, idx=[8, 9])
    m1 = _attempt(1, ["timeout", "timeout"], idx=[8, 9])
    m2 = _attempt(2, ["correct", "correct"], idx=[8, 9])
    assert P.attempts_curve([a0, a0_e1, a0_e2]) == [2, 1, 0]
    result = P.analyze([a0, a0_e1, a0_e2, m1, m2], n_total=10, n_orders=50, n_boot=20)
    assert result["failure_set_size"] == 2
    assert result["residue_natural_order"] == [2, 2, 0]
    assert result["smallest_k"] == 3
    assert result["residue_mean"][0] == 2 and result["residue_mean"][-1] == 0
    assert result["attempts_curve"] == [2, 1, 0]
    lo, hi = result["residue_ci95"]
    assert all(l <= h for l, h in zip(lo, hi))


def test_random_orders_start_with_an_anchor():
    firsts = {0: _attempt(0, ["correct"], full=True), 1: _attempt(1, ["correct"]),
              2: _attempt(2, ["correct"])}
    for order in P.random_orders(firsts, 20, random.Random(0)):
        assert order[0].full and sorted(a.model for a in order) == [0, 1, 2]


def test_ldt_forward_flops():
    # One layer, one loop, 2 tokens (1 cell + CLS), dim 4, ffn 16:
    # 2 * (4*16*2 + 2*2*2*4 + 2*4*16*2) = 2 * (128 + 32 + 256)
    assert ldt_forward_flops(4, 1, 1, 1, 4.0, True) == 2 * (128 + 32 + 256)
    assert ldt_forward_flops(128, 4, 16, 81) == 16 * ldt_forward_flops(128, 4, 1, 81)


def test_figures_and_main_tables_render(tmp_path):
    from paper.figures import fig2_sudoku_compute as fig2, fig3_maze_ksweep as fig3
    from paper.tables import table1_sudoku, table3_maze

    records = [_record(tmp_path, f"steps_{x}", s, ["correct"] * 9 + ["timeout"], x=x,
                       gpu_seconds=10.0, train_seconds=600.0)
               for x in (500, 4000) for s in (0, 1)]
    curves = fig2.percentile_curves(records, root=tmp_path)
    assert list(curves) == [500, 4000]
    assert fig2.plot(curves, fig2.solved_forwards(records[-1], tmp_path))[:4] == b"%PDF"
    assert fig3.plot(fig3.k_curve(records))[:4] == b"%PDF"

    baseline = {"label": "TRM", "result": {"n": 10000, "exact_accuracy": 0.79, "wrong": 2100,
                                           "exact_accuracy_ci": [0.78, 0.80],
                                           "gpu_seconds": 50.0}}
    tex = table1_sudoku.build(records, [baseline], {"smallest_k": 7, "n_total": 422786})
    assert "TRM & 10,000 & 79.0 [78.0, 80.0]" in tex.to_latex()
    assert "LDT portfolio, 7 models" in tex.to_latex()

    maze = [dict(r, label="LDT, K=1") for r in records if r["entry"] == "steps_500"]
    sweep = [{"step": 10, "n": 1000, "exact": 700, "lenient": 800},
             {"step": 20, "n": 1000, "exact": 720, "lenient": 810}]
    tex = table3_maze.build(maze, [{"label": "HRM", "result": sweep}], "last", root=tmp_path)
    assert "HRM (step 20) & 1 & 81.0 & 72.0 & 190" in tex.to_latex()
