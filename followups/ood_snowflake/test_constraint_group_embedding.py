"""Focused tests for opt-in Snowflake constraint-group token features."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.snowflake.data import (
    MAX_CONSTRAINT_GROUPS,
    SnowflakeConfig,
    SnowflakeDataset,
    cell_to_grid_idx,
    constraint_group_features,
    puzzle_to_state,
    random_translate,
)
from experiments.snowflake.train import N_CHANNELS, _build_state
from lattice_diffusion.models.looped_transformer import (
    LoopedTransformerConfig,
    PowersetModel,
)


def _record() -> dict:
    # Three cells are enough to test overlapping memberships.  Geometry uses
    # real covering-grid slots; the feature builder intentionally does not
    # require six-cell constraints.
    return {
        "id": 0,
        "n": 1,
        "code": "TEST",
        "puzzle": [7, 7, 7],
        "solution": [1, 2, 3],
        "givens": 0,
        "topology": {
            "n_cells": 3,
            "hex_coords": [{"q": 0, "r": 0}],
            "cell_positions": {
                "0": {"q": 0, "r": 0, "direction": "NE"},
                "1": {"q": 0, "r": 0, "direction": "E"},
                "2": {"q": 1, "r": 0, "direction": "NW"},
            },
            "constraints": [
                {"cells": [0, 1]},
                {"cells": [1, 2]},
            ],
        },
    }


def test_group_features_encode_shared_membership_with_random_ids():
    rec = _record()
    features = constraint_group_features(
        rec, MAX_CONSTRAINT_GROUPS, np.random.default_rng(7)
    )
    positions = rec["topology"]["cell_positions"]
    grid_indices = [
        cell_to_grid_idx(p["q"], p["r"], p["direction"])
        for p in positions.values()
    ]
    active = features[grid_indices]

    assert active.shape == (3, MAX_CONSTRAINT_GROUPS)
    assert active.sum(axis=1).tolist() == [1.0, 2.0, 1.0]
    # Cell 1 shares one group with each neighbor, while cells 0 and 2 do not
    # share a group directly.
    assert float((active[0] * active[1]).sum()) == 1.0
    assert float((active[1] * active[2]).sum()) == 1.0
    assert float((active[0] * active[2]).sum()) == 0.0


def test_group_id_augmentation_uses_full_vocabulary():
    rec = _record()
    used: set[int] = set()
    layouts: set[tuple[int, ...]] = set()
    for seed in range(200):
        features = constraint_group_features(
            rec, MAX_CONSTRAINT_GROUPS, np.random.default_rng(seed)
        )
        ids = tuple(np.where(features.any(axis=0))[0].tolist())
        layouts.add(ids)
        used.update(ids)

    assert len(layouts) > 100
    assert used == set(range(MAX_CONSTRAINT_GROUPS))


def test_translation_moves_group_features_with_cells():
    rec = _record()
    x, y, mask = puzzle_to_state(rec)
    groups = constraint_group_features(
        rec, MAX_CONSTRAINT_GROUPS, np.random.default_rng(3)
    )
    original_counts = sorted(groups[mask].sum(axis=1).tolist())

    translated = random_translate(
        x, y, mask, np.random.default_rng(0), group_features=groups
    )
    x_t, y_t, mask_t, groups_t = translated

    assert np.array_equal(groups_t.any(axis=1), mask_t)
    assert sorted(groups_t[mask_t].sum(axis=1).tolist()) == original_counts
    assert np.count_nonzero(x_t) == np.count_nonzero(x)
    assert np.count_nonzero(y_t) == np.count_nonzero(y)


def test_dataset_opt_in_appends_groups_without_changing_default(tmp_path):
    path = tmp_path / "snowflake.json"
    path.write_text(json.dumps([_record()]))
    common = dict(
        data_path=str(path),
        n_puzzles=1,
        batch_size=1,
        prefetch_batches=1,
        zero_hint_weight=1.0,
        correct_hint_weight=0.0,
        error_hint_weight=0.0,
    )

    default_ds = SnowflakeDataset(SnowflakeConfig(**common))
    default_batch = default_ds.next_batch()
    default_ds.close()
    assert len(default_batch) == 4

    group_ds = SnowflakeDataset(SnowflakeConfig(
        **common, constraint_group_vocab=MAX_CONSTRAINT_GROUPS
    ))
    group_batch = group_ds.next_batch()
    group_ds.close()
    assert len(group_batch) == 5
    x, _, mask, _, groups = group_batch
    state = _build_state(x.float(), mask.bool(), groups.float())
    assert state.shape == (1, 150, N_CHANNELS + MAX_CONSTRAINT_GROUPS)
    assert torch.equal(state[..., :6], x.float())
    assert torch.equal(state[..., 6], mask.float())
    assert torch.equal(state[..., 7:], groups.float())


def test_too_few_group_ids_fails_loudly():
    with np.testing.assert_raises_regex(
        ValueError, "2 constraint groups.*only 1 group IDs"
    ):
        constraint_group_features(_record(), 1, np.random.default_rng(0))


def test_model_accepts_group_augmented_state():
    cfg = LoopedTransformerConfig(
        dim=16,
        seq_len=150,
        n_channels=N_CHANNELS + MAX_CONSTRAINT_GROUPS,
        grid_rows=15,
        grid_cols=10,
        num_layers=1,
        n_heads=2,
        n_loops=1,
        cls_token=True,
        dropout=0.0,
        attn_dropout=0.0,
        ffn_dropout=0.0,
    )
    model = PowersetModel(cfg)
    state = torch.zeros(2, 150, cfg.n_channels)
    out = model(state, use_final=True)
    assert out["bce"].shape == state.shape
    assert out["softmax"].shape == state.shape
    assert out["conflict"].shape == (2, 1)
