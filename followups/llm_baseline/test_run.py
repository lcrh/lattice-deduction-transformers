import json
import tempfile
import unittest
from pathlib import Path

from followups.llm_baseline.run import (
    _checkpoint_dirs,
    make_messages,
    parse_solution,
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


if __name__ == "__main__":
    unittest.main()
