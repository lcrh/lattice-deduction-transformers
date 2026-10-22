import dataclasses

import numpy as np
import pytest
import torch

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig, SudokuExtremeDataset
from lattice_diffusion.eval import indices as index_lists


def test_named_lists_match_recorded_checksums():
    test = index_lists.load("sudoku_test_perm", split_size=422_786)
    assert len(test) == 422_786 and len(set(test.indices.tolist())) == 422_786
    val = index_lists.load("sudoku_val")
    assert len(val) == 10_000 and val.split == "train"


def test_split_size_mismatch_fails():
    with pytest.raises(ValueError):
        index_lists.load("sudoku_test_perm", split_size=1000)


def test_explicit_list_name_depends_on_contents():
    a = index_lists.explicit("sudoku", "test", [3, 1, 2])
    assert a.name == index_lists.explicit("sudoku", "test", [3, 1, 2]).name
    assert a.name != index_lists.explicit("sudoku", "test", [1, 2, 3]).name


@pytest.fixture
def small_train_cache(tmp_path, monkeypatch):
    n = 500
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, 2, (n, 81, 9), dtype=np.uint8))
    torch.save({"x": x, "y": x.clone(), "n": n}, tmp_path / "sudoku_extreme_train.pt")
    spec = index_lists.IndexListSpec(task="sudoku", split="train", split_size=n, seed=1,
                                     length=50, sha256="")
    spec = dataclasses.replace(spec, sha256=index_lists.digest(index_lists.generate(spec)))
    monkeypatch.setitem(index_lists.SPECS, "sudoku_val", spec)
    return tmp_path, set(index_lists.generate(spec).tolist())


@pytest.mark.parametrize("n_puzzles", [None, 100])
def test_training_pool_excludes_val(small_train_cache, n_puzzles):
    cache_dir, val = small_train_cache
    ds = SudokuExtremeDataset(SudokuExtremeConfig(cache_dir=str(cache_dir), n_puzzles=n_puzzles,
                                                  batch_size=4))
    try:
        pool = set(ds.pool_indices.tolist())
        assert not pool & val
        assert len(pool) == (n_puzzles or 450)
        assert ds.questions.shape[0] == len(pool)
    finally:
        ds.close()


def test_sudoku_holdout_avoids_validation_and_portfolio_training_pools():
    import numpy as np

    from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig, select_pool
    from lattice_diffusion.eval import indices as index_lists

    n = 3_831_994
    holdout = index_lists.load("sudoku_holdout", split_size=n).indices    # checks the digest
    assert len(holdout) == len(np.unique(holdout)) == 100_000
    assert not np.isin(holdout, index_lists.load("sudoku_val", split_size=n).indices).any()
    for seed in index_lists.PORTFOLIO_DATA_SEEDS:
        pool, _ = select_pool(n, SudokuExtremeConfig(n_puzzles=1000, seed=seed))
        assert not np.isin(holdout, pool).any()
