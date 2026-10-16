"""The collected-results CSVs (paper/results/<manifest>.csv): location and column types."""

from __future__ import annotations

import csv
from pathlib import Path

from paper.manifest import REPO

RESULTS = REPO / "paper" / "results"

# Columns written by paper.launch.collect that hold numbers.
NUMERIC_COLUMNS = frozenset({
    "seed", "eval_seed", "x", "n", "correct", "wrong", "timeout", "accuracy", "acc_lo", "acc_hi",
    "wrong_rate_upper", "failure_rate", "calls_mean", "calls_p50", "calls_p75", "calls_p90",
    "calls_p95", "calls_correct_mean", "unsound_rate", "gpu_seconds", "startup_seconds",
    "wall_seconds", "n_evaluated", "train_seconds", "model_dim", "model_layers", "model_loops",
    "model_heads", "model_seq_len", "model_ffn_mult", "model_channels", "train_steps",
    "train_batch", "solve_chains", "solve_max_rounds",
})
_FLOAT_COLUMNS = {"accuracy", "x"}


def parse_cell(key: str, value: str | None):
    """A CSV cell back as None, bool, int, float or str."""
    if value is None or value == "":
        return None
    if key in NUMERIC_COLUMNS:
        try:
            f = float(value)
        except ValueError:
            return value
        return int(f) if f.is_integer() and key not in _FLOAT_COLUMNS else f
    if value in ("True", "False"):
        return value == "True"
    return value


def read_results(manifest: str, results_dir: Path = RESULTS) -> list[dict]:
    with (results_dir / f"{manifest}.csv").open() as fh:
        return [{k: parse_cell(k, v) for k, v in row.items()} for row in csv.DictReader(fh)]
