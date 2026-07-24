import json
import tempfile
import unittest
from pathlib import Path

from followups.llm_baseline.run import (
    _checkpoint_dirs,
    _select_eval_checkpoints,
    apply_n_blanks,
    estimate_pass_at_k,
    make_messages,
    parse_solution,
    pass_at_k_curve,
)


class QwenSudokuExperimentTest(unittest.TestCase):
    def test_parse_solution_is_strict(self) -> None:
        solution = "123456789" * 9
        self.assertEqual(parse_solution(f"\n{solution}\n"), solution)
        self.assertIsNone(parse_solution(f"answer: {solution}"))
        self.assertIsNone(parse_solution("0" + solution[1:]))
        self.assertIsNone(parse_solution(solution[:-1]))

    def test_training_answer_is_only_in_assistant_message(self) -> None:
        question = "0" * 81
        answer = "123456789" * 9
        inference = make_messages(question)
        training = make_messages(question, answer)
        self.assertEqual(len(inference), 2)
        self.assertEqual(training[:-1], inference)
        self.assertEqual(training[-1], {"role": "assistant", "content": answer})

    def test_checkpoints_are_sorted_by_recorded_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step, epoch in [(30, 3.0), (10, 1.0), (20, 2.0)]:
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"epoch": epoch})
                )
            self.assertEqual(
                [path.name for path in _checkpoint_dirs(root)],
                ["checkpoint-10", "checkpoint-20", "checkpoint-30"],
            )

    def test_pass_at_k_curve_from_thirty_two_samples(self) -> None:
        self.assertEqual(estimate_pass_at_k(n=32, c=0, k=1), 0.0)
        self.assertEqual(estimate_pass_at_k(n=32, c=32, k=16), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(n=32, c=1, k=1), 1.0 / 32.0)
        curve = pass_at_k_curve(n=32, c=4)
        self.assertEqual(
            list(curve),
            ["pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8", "pass_at_16", "pass_at_32"],
        )
        self.assertLess(curve["pass_at_1"], curve["pass_at_2"])
        self.assertLess(curve["pass_at_2"], curve["pass_at_4"])
        self.assertEqual(curve["pass_at_32"], 1.0)

    def test_apply_n_blanks_is_exact_and_deterministic(self) -> None:
        answer = "123456789" * 9
        puzzle = apply_n_blanks(answer, n_blanks=2, seed=7, puzzle_index=3)
        self.assertEqual(puzzle.count("0"), 2)
        self.assertEqual(len(puzzle), 81)
        for q, a in zip(puzzle, answer):
            self.assertTrue(q == "0" or q == a)
        self.assertEqual(puzzle, apply_n_blanks(answer, n_blanks=2, seed=7, puzzle_index=3))
        self.assertNotEqual(puzzle, apply_n_blanks(answer, n_blanks=2, seed=7, puzzle_index=4))

    def test_select_eval_checkpoints_keeps_even_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step, epoch in [(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)]:
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"epoch": epoch})
                )
            selected = _select_eval_checkpoints(_checkpoint_dirs(root), eval_every_epochs=2)
            self.assertEqual(
                [path.name for path in selected],
                ["checkpoint-20", "checkpoint-40"],
            )


if __name__ == "__main__":
    unittest.main()
