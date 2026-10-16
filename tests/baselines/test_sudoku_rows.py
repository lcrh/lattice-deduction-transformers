import json

import numpy as np
import pytest

from baselines.hrm_trm import sudoku_rows


def _tokens(digits):
    """Digits 0..9 (0 = blank) -> upstream tokens (1 = blank, 2..10 = 1..9)."""
    return np.asarray(digits, dtype=np.int64) + 1


def test_to_digits_maps_pad_and_blank_to_zero():
    assert sudoku_rows.to_digits(np.array([0, 1, 2, 10])).tolist() == [0, 0, 1, 9]


def test_real_rows_drops_pad_rows():
    inputs = np.array([[1, 2], [0, 0], [3, 4]])
    assert sudoku_rows.real_rows(inputs).tolist() == [True, False, True]


def test_subset_test_split_reorders_and_updates_metadata(tmp_path):
    n = 6
    inputs = np.arange(n * 81).reshape(n, 81) % 10 + 1
    labels = inputs + 100
    np.save(tmp_path / "all__inputs.npy", inputs)
    np.save(tmp_path / "all__labels.npy", labels)
    (tmp_path / "dataset.json").write_text(json.dumps({"total_groups": n, "total_puzzles": n}))

    full = sudoku_rows.subset_test_split(str(tmp_path), [4, 1, 3], expected_n=n)

    assert np.array_equal(full, inputs)
    assert np.array_equal(np.load(tmp_path / "all__inputs.npy"), inputs[[4, 1, 3]])
    assert np.array_equal(np.load(tmp_path / "all__labels.npy"), labels[[4, 1, 3]])
    assert np.load(tmp_path / "all__group_indices.npy").tolist() == [0, 1, 2, 3]
    assert np.load(tmp_path / "all__puzzle_indices.npy").tolist() == [0, 1, 2, 3]
    meta = json.loads((tmp_path / "dataset.json").read_text())
    assert meta == {"total_groups": 3, "total_puzzles": 3}


def test_subset_test_split_rejects_wrong_split_size(tmp_path):
    np.save(tmp_path / "all__inputs.npy", np.ones((3, 81)))
    np.save(tmp_path / "all__labels.npy", np.ones((3, 81)))
    with pytest.raises(ValueError):
        sudoku_rows.subset_test_split(str(tmp_path), [0], expected_n=4)


def test_build_rows_exact_match_and_dataset_indices():
    solution = [((i // 9) * 3 + i // 27 + i % 9) % 9 + 1 for i in range(81)]
    labels = np.stack([_tokens(solution), _tokens(solution)])
    preds = labels.copy()
    preds[1, 0] = labels[1, 0] % 10 + 2 if labels[1, 0] < 10 else 2
    rows = sudoku_rows.build_rows([17, 5], labels, preds)
    assert [(r["idx"], r["pos"], r["status"]) for r in rows] == [(17, 0, "correct"), (5, 1, "wrong")]
    assert rows[0]["solution"] == "".join(map(str, solution))


def test_summarize_and_interval():
    rows = [{"status": "correct"}] * 9 + [{"status": "wrong"}]
    s = sudoku_rows.add_interval(sudoku_rows.summarize(rows, model="x"))
    assert (s["n"], s["correct"], s["wrong"], s["model"]) == (10, 9, 1, "x")
    lo, hi = s["exact_accuracy_ci"]
    # Clopper-Pearson 95% for 9/10.
    assert lo == pytest.approx(0.5550, abs=1e-3)
    assert hi == pytest.approx(0.9975, abs=1e-3)


def test_interval_at_the_boundaries():
    all_right = sudoku_rows.add_interval(sudoku_rows.summarize([{"status": "correct"}] * 20))
    assert all_right["exact_accuracy_ci"][1] == 1.0
    all_wrong = sudoku_rows.add_interval(sudoku_rows.summarize([{"status": "wrong"}] * 20))
    assert all_wrong["exact_accuracy_ci"][0] == 0.0


def test_first_n_of_test_perm_is_ldt_prefix():
    from lattice_diffusion.eval import indices

    assert sudoku_rows.first_n_of_test_perm(5) == indices.load("sudoku_test_perm").indices[:5].tolist()


def test_chunks_resume_and_merge_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(sudoku_rows, "CHUNK_SIZE", 3)
    indices = [40, 10, 30, 20, 50, 60, 70]
    parts = sudoku_rows.chunks(indices)
    assert [(c, start, len(idx)) for c, start, idx in parts] == [(0, 0, 3), (1, 3, 3), (2, 6, 1)]
    out = str(tmp_path)
    for c, start, idx in reversed(parts):
        assert c not in sudoku_rows.done_chunks(out)
        rows = [{"idx": i, "pos": start + k, "status": "correct" if i % 20 else "wrong"}
                for k, i in enumerate(idx)]
        sudoku_rows.write_chunk(out, c, rows, {"model": "x", "eval_seconds": 1.5})
    assert sudoku_rows.done_chunks(out) == {0, 1, 2}
    summary = sudoku_rows.merge_chunks(out, len(parts))
    assert summary["n"] == 7 and summary["correct"] == 4 and summary["gpu_seconds"] == 4.5
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [r["idx"] for r in rows] == indices
