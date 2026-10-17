"""Per-puzzle Sudoku-Extreme results for the HRM/TRM baselines.

The baselines are evaluated on the first N puzzles of `sudoku_test_perm`
(lattice_diffusion.eval.indices), the same ordering LDT's sequential evals
walk. Both upstream repos build the full test split with
`build_sudoku_dataset.py`, which reads `test.csv` in file order with no
filtering or augmentation, so row i of their test set is dataset index i in
our `sudoku_extreme_test.pt` cache (verified by `check_alignment` when the
cache is mounted). `subset_test_split` then rewrites their test split to the
requested indices, in order.

Rows use the field names of `lattice_diffusion/eval/records.py`
(`idx`, `pos`, `status`, `solution`). HRM/TRM always emit a full grid, so
`status` is "correct" (exact match) or "wrong"; there are no timeouts.
Search-specific fields (rounds, calls, forwards, n_resets) and `verified` are
not recorded.

Evaluation is resumable: the index list is cut into chunks of CHUNK_SIZE,
each chunk is written to `<out_dir>/chunks/` when it finishes, and a rerun
evaluates only the missing chunks before merging. `out_dir` is keyed by the
checkpoint and batch size, so a different model never reuses another's chunks.

Upstream token encoding: 0 = pad, 1 = blank, 2..10 = digits 1..9.
"""

import hashlib
import json
import os

import numpy as np

N_TEST = 422_786
CHUNK_SIZE = 1024


def to_digits(tokens: np.ndarray) -> np.ndarray:
    """Upstream tokens -> digits 0..9 (0 = blank / pad / invalid)."""
    d = tokens.astype(np.int64) - 1
    return np.where((d >= 1) & (d <= 9), d, 0)


def real_rows(inputs: np.ndarray) -> np.ndarray:
    """Mask out the pad rows the upstream loaders append to the last batch."""
    return (inputs != 0).all(axis=1)


def check_alignment(input_tokens: np.ndarray, cache_path: str) -> bool | None:
    """Whether every upstream test input equals our cached puzzle at its index (None: no cache)."""
    if not os.path.exists(cache_path):
        return None
    import torch

    x = torch.load(cache_path, map_location="cpu", weights_only=True)["x"]
    single = x.sum(dim=-1) == 1
    ours = ((x.argmax(dim=-1) + 1) * single).numpy()
    theirs = to_digits(input_tokens)
    return ours.shape == theirs.shape and bool((ours == theirs).all())


def subset_test_split(test_dir: str, indices: list[int], expected_n: int = N_TEST) -> np.ndarray:
    """Cut an upstream test split down to `indices`, in order.

    Returns the full-split inputs from before the rewrite, for the alignment check.
    """
    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_dir, "all__labels.npy"))
    if len(inputs) != expected_n:
        raise ValueError(f"expected {expected_n} test puzzles, got {len(inputs)}")
    idx = np.asarray(indices, dtype=np.int64)
    n = len(idx)
    arrays = {
        "inputs": inputs[idx],
        "labels": labels[idx],
        "puzzle_identifiers": np.zeros(n, dtype=np.int32),
        "puzzle_indices": np.arange(n + 1, dtype=np.int32),
        "group_indices": np.arange(n + 1, dtype=np.int32),
    }
    for k, v in arrays.items():
        np.save(os.path.join(test_dir, f"all__{k}.npy"), v)
    meta_path = os.path.join(test_dir, "dataset.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["total_groups"] = n
    if "total_puzzles" in meta:
        meta["total_puzzles"] = n
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return inputs


def build_rows(indices: list[int], label_tokens: np.ndarray,
               pred_tokens: np.ndarray, start: int = 0) -> list[dict]:
    """One row per evaluated puzzle, in index-list order; `start` is the first row's position."""
    if len(indices) != len(label_tokens):
        raise ValueError(f"{len(indices)} indices but {len(label_tokens)} predictions")
    correct = (pred_tokens == label_tokens).all(axis=1)
    pred_digits = to_digits(pred_tokens)
    return [
        {
            "idx": int(dataset_idx),
            "pos": start + k,
            "status": "correct" if correct[k] else "wrong",
            "solution": "".join(map(str, pred_digits[k])),
        }
        for k, dataset_idx in enumerate(indices)
    ]


def out_dir(model: str, n: int, checkpoint: str, batch_size: int) -> str:
    """Volume directory for one baseline eval, keyed by what changes its rows."""
    key = json.dumps({"checkpoint": checkpoint, "batch_size": batch_size}, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"/ldt_ckpt/eval/baselines/{model}/sudoku_test_perm_n{n}-{digest}"


def chunks(indices: list[int]) -> list[tuple[int, int, list[int]]]:
    """(chunk id, start position, indices) for each CHUNK_SIZE slice of `indices`."""
    return [(c, start, indices[start:start + CHUNK_SIZE])
            for c, start in enumerate(range(0, len(indices), CHUNK_SIZE))]


def _chunk_paths(out_dir: str, chunk: int) -> tuple[str, str]:
    base = os.path.join(out_dir, "chunks", f"{chunk:04d}")
    return base + ".jsonl", base + ".json"


def write_chunk(out_dir: str, chunk: int, rows: list[dict], info: dict) -> None:
    """Save one chunk's rows, then its info (whose presence marks the chunk done)."""
    rows_path, info_path = _chunk_paths(out_dir, chunk)
    os.makedirs(os.path.dirname(rows_path), exist_ok=True)
    with open(rows_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with open(info_path, "w") as fh:
        json.dump(info, fh)


def done_chunks(out_dir: str) -> set[int]:
    chunk_dir = os.path.join(out_dir, "chunks")
    if not os.path.isdir(chunk_dir):
        return set()
    return {int(name.removesuffix(".json")) for name in os.listdir(chunk_dir)
            if name.endswith(".json")}


def merge_chunks(out_dir: str, n_chunks: int, **extra) -> dict:
    """Merge every chunk into rows.jsonl and summary.json; returns the summary."""
    rows, infos = [], []
    for chunk in range(n_chunks):
        rows_path, info_path = _chunk_paths(out_dir, chunk)
        with open(rows_path) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
        with open(info_path) as fh:
            infos.append(json.load(fh))
    rows.sort(key=lambda r: r["pos"])
    seconds = sum(info.pop("eval_seconds") for info in infos)
    if any(info != infos[0] for info in infos):
        raise RuntimeError(f"chunks in {out_dir} come from different runs: "
                           f"{[i for i in infos if i != infos[0]][0]} vs {infos[0]}")
    summary = summarize(rows, **infos[0], **extra,
                        eval_seconds=seconds, gpu_seconds=seconds, n_chunks=n_chunks)
    write_outputs(out_dir, rows, summary)
    return summary


def write_outputs(out_dir: str, rows: list[dict], summary: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


def summarize(rows: list[dict], **extra) -> dict:
    n = len(rows)
    n_correct = sum(r["status"] == "correct" for r in rows)
    return {"n": n, "correct": n_correct, "wrong": n - n_correct,
            "exact_accuracy": n_correct / n if n else 0.0, **extra}


def add_interval(summary: dict, alpha: float = 0.05) -> dict:
    """Add a two-sided Clopper-Pearson interval on exact accuracy (runs locally)."""
    from lattice_diffusion.eval.stats import binomial_upper

    n, k = summary["n"], summary["correct"]
    lo = 1.0 - binomial_upper(n - k, n, alpha / 2) if k > 0 else 0.0
    hi = binomial_upper(k, n, alpha / 2)
    return {**summary, "exact_accuracy_ci": [lo, hi], "ci_alpha": alpha,
            "ci_method": "clopper-pearson"}


def first_n_of_test_perm(n: int) -> list[int]:
    """First n indices of `sudoku_test_perm` (checksum-verified). Local only."""
    from lattice_diffusion.eval import indices

    lst = indices.load("sudoku_test_perm")
    if not 0 < n <= len(lst):
        raise ValueError(f"n must be in [1, {len(lst)}]")
    return lst.indices[:n].tolist()
