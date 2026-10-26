"""Tables (paired deltas), portfolio strategies and compute formulas on synthetic rows."""

import json
from collections import Counter

import pytest

from paper import portfolio
from paper.compute import BASELINES, ldt_forward_flops, sotaku_flops, swiglu_width
from paper.tables import appendix
from paper.tables.common import eval_rule, paired_vs


def _write_rows(tmp_path, key, statuses):
    path = tmp_path / f"{key}.jsonl"
    path.write_text("".join(json.dumps({"idx": 1000 + p, "pos": p, "status": s, "calls": p + 1, "forwards_seq": 64 * (p + 1),
                                        "solution": f"s{p}"}) + "\n"
                            for p, s in enumerate(statuses)))
    return path.name


def _result_record(tmp_path, entry, seed, statuses, **kw):
    n = len(statuses)
    correct = statuses.count("correct")
    return {"entry": entry, "key": f"A1/{entry}/seed{seed}", "seed": seed, "status": "done",
            "label": kw.pop("label", entry), "group": kw.pop("group", ""), "n": n,
            "correct": correct, "accuracy": correct / n, "wrong": statuses.count("wrong"),
            "timeout": statuses.count("timeout"), "calls_mean": 3.0, "calls_per_solve": 4.0,
            "unsound_rate": 0.001,
            "stop_reason": "fixed_n",
            "rows_path": _write_rows(tmp_path, f"{entry}_{seed}", statuses), **kw}


def test_paired_delta_pools_seeds_by_puzzle(tmp_path):
    ref = [_result_record(tmp_path, "default", s, ["correct"] * 10) for s in (0, 1)]
    # Seed 0 fails puzzles 0-1, seed 1 fails puzzle 0 -> 3 of 20 pairs differ.
    arm = [_result_record(tmp_path, "arm", 0, ["timeout"] * 2 + ["correct"] * 8),
           _result_record(tmp_path, "arm", 1, ["wrong"] + ["correct"] * 9)]
    d = paired_vs(ref, arm, root=tmp_path)
    assert d.n == 20 and d.only_b == 3 and d.only_a == 0
    assert d.delta == pytest.approx(-0.15)
    assert d.lo < d.delta < d.hi


def test_appendix_table_has_reference_groups_and_deltas(tmp_path):
    records = [_result_record(tmp_path, "default", s, ["correct"] * 10, label="Default") for s in (0, 1)]
    records += [_result_record(tmp_path, "loops_1", s, ["timeout"] * 5 + ["correct"] * 5,
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


def _rows(spec):
    """Rows from (status, rounds, solution) per puzzle; calls = rounds + 1."""
    return [{"idx": i, "status": st, "rounds": rd, "calls": rd + 1, "solution": sol}
            for i, (st, rd, sol) in enumerate(spec)]


def _evals():
    # 4 puzzles, 3 models. Puzzle 3 is hard: model a times out, b answers wrong
    # (round 30), c solves it late (round 40).
    ok = lambda r: ("correct", r, "good")  # noqa: E731
    return portfolio.build_evals({
        "same_data/seed0": _rows([ok(0), ok(3), ok(12), ("timeout", 1000, "")]),
        "same_data/seed1": _rows([ok(0), ok(8), ok(2), ("wrong", 30, "bad")]),
        "diff_data_43/seed0": _rows([ok(1), ok(2), ok(4), ok(40)]),
    })


def test_portfolio_prefilter_and_resolution_rules():
    ev = _evals()
    a, b, c = 1, 2, 0       # build_evals sorts names: diff_data_43, same_data/seed0, seed1
    assert ev.names[c] == "diff_data_43/seed0"
    S = portfolio.Strategy
    # One model, full budget: b answers every puzzle, one of them wrong.
    one = portfolio.run_strategy(ev, [b], S(m=1, p=0, budget=1000, rule="first"))
    assert (one.correct, one.wrong, one.unresolved) == (3, 1, 0)
    assert one.calls == 1 + 9 + 3 + 31
    # Prefilter a at R=5 accepts puzzles 0 and 1; b then answers 2 right, 3 wrong.
    out = portfolio.run_strategy(ev, [a, b], S(m=2, p=1, budget=5, rule="first"))
    assert (out.correct, out.wrong, out.unresolved) == (3, 1, 0)
    assert out.calls == (1 + 4 + 6 + 6) + (3 + 31)     # prefilter min(calls, R+1), then b
    # Two prefilter models must agree: a and c agree on 0 and 1 within R=5;
    # puzzle 2 (a needs 12 rounds) and 3 fall to b.
    out = portfolio.run_strategy(ev, [a, c, b], S(m=3, p=2, budget=5, rule="first"))
    assert (out.correct, out.wrong) == (3, 1)
    # Majority over b and c on puzzle 3 is a 1-1 tie -> unresolved, no wrong answer.
    out = portfolio.run_strategy(ev, [a, b, c], S(m=3, p=1, budget=5, rule="majority"))
    assert (out.correct, out.wrong, out.unresolved) == (3, 0, 1)
    # Two agreeing answers needed: puzzle 3 never gets them.
    out = portfolio.run_strategy(ev, [a, b, c], S(m=3, p=0, budget=1000, rule="agree2"))
    assert (out.correct, out.wrong, out.unresolved) == (3, 0, 1)


def test_resolve_handles_colliding_wrong_answers():
    import numpy as np
    ans = np.array([[5, 0], [5, 0], [0, 5]])     # puzzle 0: two identical wrong votes
    calls = np.ones_like(ans)
    chosen, spent = portfolio._resolve(ans, calls, "majority")
    assert chosen.tolist() == [5, 0]
    chosen, spent = portfolio._resolve(ans, calls, "agree2")
    assert chosen.tolist() == [5, 0] and spent.tolist() == [2, 2]


def test_load_evals_skips_test_split_evals(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"idx": 0, "status": "correct", "rounds": 1,
                                "calls": 2, "solution": "good"}) + "\n")
    rec = lambda entry, index_list: {  # noqa: E731
        "entry": entry, "seed": 0, "status": "done", "n": 1, "rows_path": "rows.jsonl",
        "set": json.dumps({"eval.index_list": index_list} if index_list else {})}
    ev = portfolio.load_evals([rec("same_data", "sudoku_holdout"),
                               rec("same_data_test", None)], root=tmp_path)
    assert ev.names == ["same_data/seed0"]


def test_test_portfolios_use_the_first_response(tmp_path, monkeypatch):
    # 3 test puzzles, 15 models. Puzzle 1: seed 0 times out at 1,000 rounds and
    # answers wrong at 10,000; seed 1 answers the same wrong grid at 1,000.
    # Puzzle 2: seed 5 times out at both budgets.
    def rec(entry, seed, rows):
        path = tmp_path / f"{entry}-{seed}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return {"entry": entry, "seed": seed, "status": "done", "n": len(rows),
                "rows_path": path.name}
    row = lambda i, st, sol="good": {"idx": i, "status": st, "solution": sol}  # noqa: E731
    first = {s: [row(0, "correct"), row(1, "correct"), row(2, "correct")] for s in range(15)}
    first[0][1] = row(1, "timeout", "")
    first[1][1] = row(1, "wrong", "bad")
    first[5][2] = row(2, "timeout", "")
    records = [rec("same_data_test", s, rows) for s, rows in first.items()]
    hard = [row(1, "wrong", "bad"), row(2, "correct")]
    records += [rec("same_data_test_r10k", s, hard) for s in (0, 1)]
    records.append(rec("same_data_test_r10k", 5, [row(1, "correct"), row(2, "timeout", "")]))

    fails, n = portfolio.test_failures(records, root=tmp_path)
    assert n == 3 and portfolio.hard_set(fails) == [1, 2]
    by = {(r["portfolio"], r["rule"]): r
          for r in portfolio.score_test(*portfolio.test_evals(records, root=tmp_path))}
    assert by["seeds 0-4", "agree2"]["wrong"] == 1           # seeds 0 and 1 agree on "bad"
    assert by["seeds 0-4", "agree3"]["wrong"] == 0
    assert by["seeds 5-9", "first"]["unresolved"] == 0       # seed 6 answers puzzle 2
    assert by["all 15", "majority"]["accuracy"] == 1.0


def test_simulate_reports_every_strategy():
    ev = _evals()
    rows = portfolio.simulate(ev, {"all": [0, 1, 2]}, draws=5, max_m=3)
    assert {r["m"] for r in rows} == {1, 2, 3}
    best = [r for r in rows if r["m"] == 3 and r["p"] == 0 and r["rule"] == "agree2"][0]
    assert best["wrong_max"] == 0 and best["unresolved_mean"] == 1
    assert "0 wrong in every draw" in portfolio.summary_text(rows)


def test_baseline_flops_follow_the_upstream_layers():
    assert swiglu_width(512, 4) == 1536 and swiglu_width(97, 4) == 512
    mlp, att = BASELINES["trm_mlp_sudoku"], BASELINES["trm_att_sudoku"]
    ffn = 3 * 512 * 1536 * 97
    assert mlp.forward_flops() == 42 * 2 * (3 * 97 * 512 * 512 + ffn)
    assert att.forward_flops() == 42 * 2 * (4 * 512 * 512 * 97 + 2 * 97 * 97 * 512 + ffn)


def test_training_flops_match_the_paper():
    pf = {k: a.training_flops() / 1e15 for k, a in BASELINES.items()}
    assert round(pf["hrm_sudoku"]) == 1466 and round(pf["trm_mlp_sudoku"]) == 2136
    assert round(pf["eqr_sudoku"]) == 1640 and round(pf["hrm_maze"]) == 14982
    assert round(BASELINES["eqr_sudoku"].max_inference_flops() / 1e12) == 210
    assert round(sotaku_flops()["training"] / 1e15) == 1265
    assert 4 * ldt_forward_flops(128, 4, 16, 81) * 512 * 4000 / 1e15 == pytest.approx(18.7, abs=0.05)


def test_ldt_forward_flops():
    # One layer, one loop, 2 tokens (1 cell + CLS), dim 4, ffn 16:
    # 2 * (4*16*2 + 2*2*2*4 + 2*4*16*2) = 2 * (128 + 32 + 256)
    assert ldt_forward_flops(4, 1, 1, 1, 4.0, True) == 2 * (128 + 32 + 256)
    assert ldt_forward_flops(128, 4, 16, 81) == 16 * ldt_forward_flops(128, 4, 1, 81)


def test_figures_and_main_tables_render(tmp_path):
    from paper.figures import fig2_sudoku_compute as fig2, fig3_maze_ksweep as fig3
    from paper.tables import table1_sudoku, table3_maze

    records = [_result_record(tmp_path, f"steps_{x}", s, ["correct"] * 9 + ["timeout"], x=x,
                       gpu_seconds=10.0, n_evaluated=10, train_seconds=600.0,
                       train_post_compile_seconds=540.0, train_batch=512,
                       calls_correct_mean=5.0)
               for x in (500, 4000) for s in (0, 1)]
    curves = fig2.percentile_curves(records, root=tmp_path)
    assert list(curves) == [500, 4000]
    assert curves[4000]["examples"] == 4000 * 512
    densities = {4000: fig2.solved_forwards(records[-1], tmp_path)}
    budget = fig2.budget_curves([{"correct": Counter({0: 5, 3: 4}), "wrong": Counter({7: 1}),
                                  "timeout": Counter({-1: 2})}])
    assert budget["solved"][0, 0] == 5 / 12 and budget["wrong"][0, -1] == 1 / 12
    assert fig2.plot(curves, densities, budget)[:4] == b"%PDF"
    assert fig3.plot(fig3.k_curve(records))[:4] == b"%PDF"

    baseline = {"label": "TRM-MLP (released)",
                "result": {"n": 10000, "exact_accuracy": 0.79, "wrong": 2100}}
    rows = [dict(r, entry="steps4k_train_noaug") for r in records if r["entry"] == "steps_4000"]
    tex = table1_sudoku.build(rows, [baseline]).to_latex()
    assert "TRM-MLP (released) & 79.00 & 21.00000" in tex
    assert "LDT, 4K train steps & 90.000 & 0.00000" in tex      # pooled over seeds

    # Maze: each seed's best of its final checkpoint and step snapshots.
    maze = [dict(r, entry="k1", train_steps=20000) for r in records if r["entry"] == "steps_500"]
    maze += [_result_record(tmp_path, "k1_snap_step10000", s, ["correct"] * 10) for s in (0, 1)]
    sweep = [{"step": 10, "n": 1000, "exact": 700, "lenient": 800},
             {"step": 20, "n": 1000, "exact": 720, "lenient": 810}]
    tex = table3_maze.build(maze, [{"label": "HRM", "result": sweep}], "last", root=tmp_path).to_latex()
    assert "HRM & 1 & 81.0 & 72.0 & 190 & 0 & 20" in tex
    assert "LDT, K=1 & 2 &" in tex and "& 10,000, 10,000" in tex
