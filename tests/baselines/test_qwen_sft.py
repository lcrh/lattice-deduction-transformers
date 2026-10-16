import json

import pytest

from baselines.qwen_sft.run import (
    PASS_KS, apply_n_blanks, checkpoint_dirs, estimate_pass_at_k, make_messages, make_rows,
    parse_solution, pass_at_k_curve, select_checkpoints, summarize_records,
)

SOLUTION = "123456789" * 9


def test_parse_solution_is_strict():
    assert parse_solution(f"\n{SOLUTION}\n") == SOLUTION
    assert parse_solution(f"answer: {SOLUTION}") is None
    assert parse_solution("0" + SOLUTION[1:]) is None
    assert parse_solution(SOLUTION[:-1]) is None


def test_training_answer_is_only_in_assistant_message():
    inference = make_messages("0" * 81)
    training = make_messages("0" * 81, SOLUTION)
    assert len(inference) == 2
    assert training[:-1] == inference
    assert training[-1] == {"role": "assistant", "content": SOLUTION}


def _make_checkpoints(root, steps_epochs):
    for step, epoch in steps_epochs:
        ckpt = root / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({"epoch": epoch}))


def test_checkpoints_are_sorted_by_recorded_epoch(tmp_path):
    _make_checkpoints(tmp_path, [(30, 3.0), (10, 1.0), (20, 2.0)])
    assert [p.name for p in checkpoint_dirs(tmp_path)] == [
        "checkpoint-10", "checkpoint-20", "checkpoint-30"]


def test_select_checkpoints_keeps_every_nth_epoch(tmp_path):
    _make_checkpoints(tmp_path, [(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)])
    selected = select_checkpoints(checkpoint_dirs(tmp_path), every_epochs=2)
    assert [p.name for p in selected] == ["checkpoint-20", "checkpoint-40"]


def test_pass_at_k_curve_from_thirty_two_samples():
    assert estimate_pass_at_k(n=32, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=32, c=32, k=16) == 1.0
    assert estimate_pass_at_k(n=32, c=1, k=1) == pytest.approx(1 / 32)
    curve = pass_at_k_curve(n=32, c=4)
    assert list(curve) == [f"pass_at_{k}" for k in PASS_KS]
    assert curve["pass_at_1"] < curve["pass_at_2"] < curve["pass_at_4"]
    assert curve["pass_at_32"] == 1.0


def test_apply_n_blanks_is_exact_and_deterministic():
    puzzle = apply_n_blanks(SOLUTION, n_blanks=2, seed=7, puzzle_index=3)
    assert puzzle.count("0") == 2 and len(puzzle) == 81
    assert all(q in ("0", a) for q, a in zip(puzzle, SOLUTION))
    assert puzzle == apply_n_blanks(SOLUTION, n_blanks=2, seed=7, puzzle_index=3)
    assert puzzle != apply_n_blanks(SOLUTION, n_blanks=2, seed=7, puzzle_index=4)


def test_make_rows_keeps_order_and_normalizes_blanks():
    question = "." + SOLUTION[1:]
    dataset = {5: {"question": question, "answer": SOLUTION},
               2: {"question": SOLUTION, "answer": SOLUTION}}
    natural = make_rows(dataset, [5, 2], n_blanks=None)
    assert [r["idx"] for r in natural] == [5, 2]
    assert natural[0]["question"] == "0" + SOLUTION[1:]
    blanked = make_rows(dataset, [5, 2], n_blanks=3)
    assert all(r["question"].count("0") == 3 for r in blanked)


def test_summary_counts_and_mean_pass_at_k():
    records = [
        {"n_correct": 0, "n_malformed": 1, **pass_at_k_curve(4, 0, (1, 2, 4))},
        {"n_correct": 4, "n_malformed": 0, **pass_at_k_curve(4, 4, (1, 2, 4))},
    ]
    s = summarize_records(records, samples_per_puzzle=4)
    assert s["pass_at_1"] == 0.5 and s["puzzles_passed"] == 1
    assert (s["correct_samples"], s["malformed_samples"], s["total_samples"]) == (4, 1, 8)
