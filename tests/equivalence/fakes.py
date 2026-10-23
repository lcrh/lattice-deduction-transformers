"""Synthetic data and tiny models for CPU equivalence tests.

Everything is seeded so the reference fixtures (captured from the code
before a refactor) and the refactored code see identical inputs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from ldt.eval import indices as index_lists
from ldt.models.looped_transformer import LoopedTransformerConfig, PowersetModel

FIXTURES = Path(__file__).parent / "fixtures"


def tiny_model_cfg(**kw) -> LoopedTransformerConfig:
    base = dict(dim=16, num_layers=1, n_heads=2, n_loops=2, cls_token=True)
    base.update(kw)
    return LoopedTransformerConfig(**base)


def tiny_model(seed: int = 0, **kw) -> PowersetModel:
    torch.manual_seed(seed)
    return PowersetModel(tiny_model_cfg(**kw))


def sudoku_grids(n: int, rng: np.random.Generator) -> np.ndarray:
    """[n, 81] valid solved grids (digits 1..9) from a shuffled pattern grid."""
    base = np.array([[(r * 3 + r // 3 + c) % 9 for c in range(9)] for r in range(9)])
    out = np.zeros((n, 81), dtype=np.int64)
    for i in range(n):
        g = rng.permutation(9)[base]
        bands = rng.permutation(3)
        rows = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
        stacks = rng.permutation(3)
        cols = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
        out[i] = (g[rows][:, cols] + 1).reshape(81)
    return out


def sudoku_tensors(n: int, n_blank: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) uint8 [n, 81, 9]: powerset puzzle with blanks, one-hot solution."""
    rng = np.random.default_rng(seed)
    grids = sudoku_grids(n, rng)
    y = np.eye(9, dtype=np.uint8)[grids - 1]
    x = y.copy()
    for i in range(n):
        k = int(rng.integers(n_blank[0], n_blank[1] + 1))
        x[i, rng.choice(81, size=k, replace=False)] = 1
    return torch.from_numpy(x), torch.from_numpy(y)


def write_sudoku_caches(cache_dir: Path, n_train: int = 300, n_test: int = 40) -> None:
    for split, n, seed in (("train", n_train, 1), ("test", n_test, 2)):
        x, y = sudoku_tensors(n, (20, 50), seed)
        torch.save({"x": x, "y": y, "n": n}, cache_dir / f"sudoku_extreme_{split}.pt")


@contextlib.contextmanager
def small_sudoku_val(n_train: int = 300, length: int = 30):
    """Point the `sudoku_val` index list at a fake train split of n_train."""
    spec = index_lists.IndexListSpec(task="sudoku", split="train", split_size=n_train,
                                     seed=7, length=length, sha256="")
    spec = dataclasses.replace(spec, sha256=index_lists.digest(index_lists.generate(spec)))
    old = index_lists.SPECS["sudoku_val"]
    index_lists.SPECS["sudoku_val"] = spec
    try:
        yield spec
    finally:
        index_lists.SPECS["sudoku_val"] = old


_DIRS = ("NW", "NE", "W", "E", "SW", "SE")


def write_snowflake_parquet(path: Path, n_rows: int, seed: int, n_cells: int = 12,
                            orders: tuple[int, ...] = (4,)) -> None:
    """Fake Snowflake rows with the generator's schema.

    Constraint groups are runs of 6 cells (not a real topology); `n` cycles
    through `orders`, and nothing else depends on it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    slots = [(q, r, d) for q in range(-2, 3) for r in range(-2, 3) for d in _DIRS]
    rows = []
    for i in range(n_rows):
        chosen = rng.choice(len(slots), size=n_cells, replace=False)
        positions = {str(c): {"q": slots[s][0], "r": slots[s][1], "direction": slots[s][2]}
                     for c, s in enumerate(chosen)}
        solution = rng.integers(1, 7, size=n_cells).tolist()
        puzzle = [7 if rng.random() < 0.6 else v for v in solution]
        constraints = [{"cells": list(range(k, min(k + 6, n_cells)))}
                       for k in range(0, n_cells, 6)]
        rows.append({
            "id": i, "n": orders[i % len(orders)], "code": f"f{i}", "puzzle": puzzle,
            "solution": solution, "givens": sum(v != 7 for v in puzzle),
            "topology": json.dumps({"n_cells": n_cells, "cell_positions": positions,
                                    "constraints": constraints}),
        })
    table = pa.table({
        "id": pa.array([r["id"] for r in rows], type=pa.int64()),
        "n": pa.array([r["n"] for r in rows], type=pa.int32()),
        "code": pa.array([r["code"] for r in rows], type=pa.string()),
        "puzzle": pa.array([r["puzzle"] for r in rows], type=pa.list_(pa.int8())),
        "solution": pa.array([r["solution"] for r in rows], type=pa.list_(pa.int8())),
        "givens": pa.array([r["givens"] for r in rows], type=pa.int32()),
        "topology": pa.array([r["topology"] for r in rows], type=pa.string()),
    })
    pq.write_table(table, path)


def random_lattice(n: int, s: int, c: int, seed: int, p_dead: float = 0.3) -> torch.Tensor:
    """[n, s, c] float lattice with every cell keeping >= 1 alive bit."""
    g = torch.Generator().manual_seed(seed)
    alive = torch.rand(n, s, c, generator=g) > p_dead
    keep = torch.randint(0, c, (n, s), generator=g)
    alive.scatter_(-1, keep.unsqueeze(-1), True)
    return alive.float()


def state_dict_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
