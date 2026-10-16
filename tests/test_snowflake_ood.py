"""Snowflake order-transfer support: training-pool selection, translation and
constraint-group inputs, generator splitting, and order-filtered evaluation."""

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lattice_diffusion import config as cfglib
from lattice_diffusion.eval import indices as index_lists
from lattice_diffusion.eval.runner import block_args, resolve_index_list, solve_block
from lattice_diffusion.eval.tasks import get_task
from experiments.registry import experiment_class
from experiments.snowflake import gen_data
from experiments.snowflake.data import (
    MAX_CONSTRAINT_GROUPS, SEQ_LEN, SnowflakeConfig, _load_puzzles, constraint_group_features,
    puzzle_to_state, random_translate, select_pool, split_name,
)
from tests.equivalence import fakes

ORDERS = (4, 5, 6, 7, 8, 9, 10)
CONFIGS = Path(__file__).parent.parent / "configs" / "snowflake"


def _puzzles(tmp: Path, n: int = 140) -> list[dict]:
    fakes.write_snowflake_parquet(tmp / "p.parquet", n, seed=5, orders=ORDERS)
    return _load_puzzles(str(tmp / "p.parquet"))


def test_default_pool_is_the_historical_subset(tmp_path):
    puzzles = _puzzles(tmp_path)
    pool = select_pool(puzzles, SnowflakeConfig(n_puzzles=40, seed=42))
    expected = np.random.default_rng(42).choice(len(puzzles), 40, replace=False)
    assert np.array_equal(pool, expected)
    assert np.array_equal(select_pool(puzzles, SnowflakeConfig(seed=42)), np.arange(len(puzzles)))


def test_order_filter_and_exact_mixin(tmp_path):
    puzzles = _puzzles(tmp_path)
    orders = np.array([p["n"] for p in puzzles])
    pool = select_pool(puzzles, SnowflakeConfig(train_orders=[4, 5, 6], n_puzzles=30))
    assert len(pool) == 30 and set(orders[pool]) <= {4, 5, 6}
    counts = {4: 10, 5: 10, 6: 8, 9: 1, 10: 1}
    pool = select_pool(puzzles, SnowflakeConfig(train_order_counts=counts, n_puzzles=30))
    got = {o: int((orders[pool] == o).sum()) for o in set(orders[pool])}
    assert got == counts and len(set(pool.tolist())) == 30
    with pytest.raises(ValueError):
        select_pool(puzzles, SnowflakeConfig(train_order_counts=counts, n_puzzles=31))
    with pytest.raises(ValueError):
        select_pool(puzzles, SnowflakeConfig(train_order_counts={4: 1000}))


def test_translation_moves_whole_hexes_and_keeps_groups(tmp_path):
    rec = _puzzles(tmp_path, 1)[0]
    x, y, mask = puzzle_to_state(rec)
    groups = constraint_group_features(rec, np.random.default_rng(0))
    for seed in range(20):
        (tx, ty, tg), tmask = random_translate([x, y, groups], mask, np.random.default_rng(seed))
        before, after = np.divmod(np.where(mask)[0], 10), np.divmod(np.where(tmask)[0], 10)
        d_row, d_col = after[0] - before[0], after[1] - before[1]
        # One rigid shift for every cell, by whole hexes (3 rows x 2 cols of slots).
        assert len(set(d_row)) == 1 and len(set(d_col)) == 1
        assert d_row[0] % 3 == 0 and d_col[0] % 2 == 0
        assert np.array_equal(tx[tmask], x[mask]) and np.array_equal(ty[tmask], y[mask])
        assert np.array_equal(tg[tmask], groups[mask])


def test_constraint_groups_get_distinct_random_ids(tmp_path):
    rec = _puzzles(tmp_path, 1)[0]
    a = constraint_group_features(rec, np.random.default_rng(0))
    b = constraint_group_features(rec, np.random.default_rng(1))
    assert a.shape == (SEQ_LEN, MAX_CONSTRAINT_GROUPS)
    n_groups = len(rec["topology"]["constraints"])
    assert (a.sum(axis=0) > 0).sum() == n_groups
    assert not np.array_equal(a, b)
    _, _, mask = puzzle_to_state(rec)
    assert (a[mask].sum(axis=1) >= 1).all() and a[~mask].sum() == 0


def _row(i, n, puzzle):
    return {"id": -1, "n": n, "code": f"c{i:04d}", "puzzle": puzzle, "solution": [1] * 6,
            "givens": 1, "topology": json.dumps({"layout": i % 3})}


def test_validation_split_leaves_test_unchanged():
    rows = [_row(i, 4 + i % 3, [i % 7 + 1, i // 7 % 7 + 1, i // 49 % 7 + 1, 7, 7, 7])
            for i in range(300)]
    without, _ = gen_data.split_rows([dict(r) for r in rows], "canonical", 0.1, 0.0, seed=0)
    with_val, _ = gen_data.split_rows([dict(r) for r in rows], "canonical", 0.1, 0.05, seed=0)
    assert [r["code"] for r in without["test"]] == [r["code"] for r in with_val["test"]]
    assert sorted(r["code"] for r in with_val["train"] + with_val["val"]) == \
        sorted(r["code"] for r in without["train"])
    assert with_val["val"]


def test_canonical_dedup_ignores_layout_varied_does_not():
    rows = [_row(i, 4, [1, 2, 3, 7, 7, 7]) for i in range(3)]   # same digits, 3 layouts
    canonical, dropped = gen_data.split_rows([dict(r) for r in rows], "canonical", 0.1, 0.05, 0)
    assert dropped == 2
    varied, dropped = gen_data.split_rows([dict(r) for r in rows], "varied", 0.1, 0.05, 0)
    assert dropped == 0 and sum(len(v) for v in varied.values()) == 3


def test_ood_configs_load_with_their_order_settings():
    cls = experiment_class("snowflake")
    leq6 = cfglib.load(cls, CONFIGS / "ood_4to6_eval7to10.yaml")
    assert leq6.data.train_orders == [4, 5, 6]
    assert leq6.eval.index_filter == {"orders": [7, 8, 9, 10]}
    for k in (12, 25, 50):
        exp = cfglib.load(cls, CONFIGS / f"ood_4to6_mixin{k}.yaml")
        counts = exp.data.train_order_counts
        assert sum(counts.values()) == exp.data.n_puzzles == 500
        assert sum(v for o, v in counts.items() if o >= 7) == k
    spec = leq6.eval_spec("/ckpt/x.pt")
    assert spec.index_filter == {"orders": [7, 8, 9, 10]} and "orders=7,8,9,10" in spec.tag


def test_group_channels_widen_the_model():
    exp = cfglib.load(experiment_class("snowflake"), None, "data.constraint_group_embed=true")
    assert exp.model.n_channels == 7 + MAX_CONSTRAINT_GROUPS


@pytest.mark.parametrize("group_embed", [False, True])
def test_order_filtered_eval_end_to_end(group_embed):
    from experiments.snowflake.config import SnowflakeExperiment
    from experiments.snowflake.train import TrainConfig, snowflake_step, train

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split, n, seed in (("train", 60, 31), ("val", 10, 33), ("test", 21, 32)):
            fakes.write_snowflake_parquet(tmp / f"snowflake_varied_{split}.parquet", n,
                                          seed=seed, orders=ORDERS)
        exp = SnowflakeExperiment(
            model=fakes.tiny_model_cfg(),
            data=SnowflakeConfig(data_dir=str(tmp), dataset="varied", n_puzzles=30,
                                 train_orders=[4, 5, 6], translate_aug=True,
                                 constraint_group_embed=group_embed),
            train=TrainConfig(steps=2, batch_size=16, step=snowflake_step(augment=True),
                              compile=False, monitor_every=1, monitor_n_puzzles=4,
                              monitor_max_rounds=2, log_every=1),
        )
        exp.eval.index_filter = {"orders": [7, 8]}
        ckpt = tmp / "sf.pt"
        train(exp, ckpt)

        task = get_task("snowflake", str(tmp))
        solve = exp.solve_dict()
        solve.update(max_rounds=3, n_chains=4, batch_size=8)
        spec = dataclasses.replace(exp.eval_spec(str(ckpt)), solve={**solve, "dropout_p": 0.05},
                                   block_size=4)
        lst = resolve_index_list(spec, task.split_size, task.filter_indices)
        assert lst.name == "snowflake_varied_test[orders=7,8]"
        assert len(lst) == 6       # 21 test rows cycle through 7 orders: 3 each of 7 and 8
        model = task.load_model(str(ckpt), 0.05, False)
        rows = solve_block(task, model, block_args(spec, lst, 0))
        assert {r["order"] for r in rows} <= {7, 8}
        assert rows == solve_block(task, model, block_args(spec, lst, 0))


def test_filter_needs_task_support():
    spec = experiment_class("snowflake")().eval_spec("/ckpt/x.pt")
    with pytest.raises(ValueError):
        resolve_index_list(spec, split_size=lambda split: 10)


def test_split_names():
    assert split_name("canonical", "test") == "test"
    assert split_name("varied", "val") == "varied_val"
    assert gen_data.split_file("varied", "val") == "snowflake_varied_val.parquet"
    assert gen_data.split_file("canonical", "train") == "snowflake_train.parquet"
    assert "snowflake_varied_test" in index_lists.SPECS
