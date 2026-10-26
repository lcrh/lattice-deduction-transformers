"""Snowflake order transfer: pool selection, translation, group inputs, splits, order filters."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ldt import config as cfglib
from ldt.eval import indices as index_lists
from ldt.eval.runner import block_args, resolve_index_list, solve_block
from ldt.eval.tasks import get_task
from experiments.registry import experiment_class
from experiments.snowflake import gen_data
from experiments.snowflake.data import (
    MAX_CONSTRAINT_GROUPS, SEQ_LEN, SnowflakeConfig, load_puzzles, constraint_group_features,
    puzzle_to_state, random_translate, select_pool, split_file, split_name,
)
from tests.equivalence import fakes

ORDERS = (4, 5, 6, 7, 8, 9, 10)
CONFIGS = Path(__file__).parent.parent / "configs" / "snowflake"


def _puzzles(tmp: Path, n: int = 140) -> list[dict]:
    fakes.write_snowflake_parquet(tmp / "p.parquet", n, seed=5, orders=ORDERS)
    return load_puzzles(tmp / "p.parquet")


def test_default_pool_is_a_seeded_subset(tmp_path):
    puzzles = _puzzles(tmp_path)
    pool = select_pool(puzzles, SnowflakeConfig(n_puzzles=40, seed=42))
    expected = np.random.default_rng(42).choice(len(puzzles), 40, replace=False)
    assert np.array_equal(pool, expected)
    assert np.array_equal(select_pool(puzzles, SnowflakeConfig(seed=42)), np.arange(len(puzzles)))


def test_order_filter(tmp_path):
    puzzles = _puzzles(tmp_path)
    orders = np.array([p["n"] for p in puzzles])
    pool = select_pool(puzzles, SnowflakeConfig(train_orders=[4, 5, 6], n_puzzles=30))
    assert len(pool) == 30 and set(orders[pool]) <= {4, 5, 6}
    with pytest.raises(ValueError):
        select_pool(puzzles, SnowflakeConfig(train_orders=[19]))


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


def _row(i, n, puzzle, solution=None):
    return {"id": -1, "n": n, "code": f"c{i:04d}", "puzzle": puzzle,
            "solution": solution or [i % 7 + 1, i // 7 % 7 + 1, i // 49 % 7 + 1, 1, 1, 1],
            "givens": 1, "topology": json.dumps({"layout": i % 3})}


def test_validation_split_leaves_test_unchanged():
    rows = [_row(i, 4 + i % 3, [i % 7 + 1, i // 7 % 7 + 1, i // 49 % 7 + 1, 7, 7, 7])
            for i in range(300)]
    without, _ = gen_data.split_rows([dict(r) for r in rows], 0.1, 0.0, seed=0)
    with_val, _ = gen_data.split_rows([dict(r) for r in rows], 0.1, 0.05, seed=0)
    assert [r["code"] for r in without["test"]] == [r["code"] for r in with_val["test"]]
    assert sorted(r["code"] for r in with_val["train"] + with_val["val"]) == \
        sorted(r["code"] for r in without["train"])
    assert with_val["val"]


def test_dedup_keeps_distinct_layouts():
    rows = [_row(i, 4, [1, 2, 3, 7, 7, 7], [1] * 6) for i in range(6)]   # same digits, 3 layouts
    splits, dropped = gen_data.split_rows([dict(r) for r in rows], 0.1, 0.05, 0)
    assert dropped == 3 and sum(len(v) for v in splits.values()) == 3


def test_variants_of_a_puzzle_share_a_split():
    rows = [_row(100 * b + v, 4, [b % 7 + 1] + [7] * v + [1] * (5 - v),
                 [b % 7 + 1, b // 7 % 7 + 1, 1, 1, 1, 1])
            for b in range(49) for v in range(4)]   # 4 given-count variants per puzzle
    for r in rows:
        r["topology"] = json.dumps({"layout": r["solution"][:2]})
    splits, _ = gen_data.split_rows([dict(r) for r in rows], 0.2, 0.1, 0)
    split_of = {}
    for name, rs in splits.items():
        for r in rs:
            assert split_of.setdefault(gen_data.split_key(r), name) == name
    assert splits["test"] and splits["train"]


def test_ood_configs_load_with_their_order_settings():
    cls = experiment_class("snowflake")
    leq6 = cfglib.load(cls, CONFIGS / "extreme_ood_4to6_eval7to10.yaml")
    assert leq6.data.train_orders == [4, 5, 6]
    assert leq6.eval.index_filter == {"orders": [7, 8, 9, 10]}
    leq8 = cfglib.load(cls, CONFIGS / "extreme_ood_6to8_eval9to10.yaml")
    assert leq8.data.train_orders == [6, 7, 8]
    assert leq8.eval.index_filter == {"orders": [9, 10]}
    spec = leq6.eval_spec("/ckpt/x.pt")
    assert spec.index_filter == {"orders": [7, 8, 9, 10]} and "orders=7,8,9,10" in spec.tag


def test_group_channels_widen_the_model():
    exp = cfglib.load(experiment_class("snowflake"), None, "data.constraint_group_embed=true")
    assert exp.model.n_channels == 7 + MAX_CONSTRAINT_GROUPS


@pytest.mark.parametrize("group_embed", [False, True])
def test_order_filtered_eval_end_to_end(group_embed, tmp_path, monkeypatch):
    tmp = tmp_path
    # The synthetic test split is tiny, so use an unpinned copy of the list.
    monkeypatch.setitem(index_lists.SPECS, "snowflake_test", dataclasses.replace(
        index_lists.SPECS["snowflake_test"], split_size=None, sha256=None))
    from experiments.snowflake.config import SnowflakeExperiment
    from experiments.snowflake.train import TrainConfig, snowflake_step, train

    for split, n, seed in (("train", 60, 31), ("val", 10, 33), ("test", 21, 32)):
        fakes.write_snowflake_parquet(split_file(str(tmp), split), n,
                                      seed=seed, orders=ORDERS)
    exp = SnowflakeExperiment(
        model=fakes.tiny_model_cfg(),
        data=SnowflakeConfig(data_dir=str(tmp), n_puzzles=30,
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
    assert lst.name == "snowflake_test[orders=7,8]"
    assert len(lst) == 6       # 21 test rows cycle through 7 orders: 3 each of 7 and 8
    model = task.load_model(str(ckpt), 0.05, False)
    rows = solve_block(task, model, block_args(spec, lst, 0)).rows
    assert {r["order"] for r in rows} <= {7, 8}
    assert rows == solve_block(task, model, block_args(spec, lst, 0)).rows


def test_filter_needs_task_support():
    spec = experiment_class("snowflake")().eval_spec("/ckpt/x.pt")
    with pytest.raises(ValueError):
        resolve_index_list(spec, split_size=lambda split: 10)


def test_split_file_names_match_the_generator(tmp_path):
    for split in ("train", "val", "test"):
        assert split_file(str(tmp_path), split).name == gen_data.split_file(split)
    assert {"snowflake_test", "snowflake_val"} <= index_lists.SPECS.keys()


def test_rating_counts_guesses():
    groups = [list(range(6)), list(range(6, 12))]
    solution = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    one_blank = [7] + solution[1:]
    assert gen_data.rating(one_blank, groups) == 0
    # A blank group has many completions; singles cannot pick one.
    assert gen_data.rating([7] * 6 + solution[6:], groups) >= 1
    with pytest.raises(ValueError):
        gen_data.rating([1, 1] + [7] * 10, groups)


def test_extreme_is_hashed_only_when_set():
    cls = experiment_class("snowflake")
    plain = cls()
    extreme = cfglib.from_dict(cls, cfglib.merge(cfglib.to_dict(plain), {"data": {"extreme": True}}))
    assert "extreme" not in plain.training_dict()["data"]
    assert extreme.training_dict()["data"]["extreme"] is True
    assert extreme.train_hash() != plain.train_hash()
    assert split_name(extreme.data, "train") == "extreme_train"
    assert split_name(plain.data, "val") == "val"
