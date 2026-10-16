"""Combine per-puzzle results of several eval runs (model portfolios).

Each run is a mapping idx -> row (see records.py). Only puzzles present in
every run are combined.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from lattice_diffusion.eval.records import read_blocks
from lattice_diffusion.eval.scheduler import Row


def load_rows(run_dir: str | Path) -> dict[int, Row]:
    """All evaluated rows of a run (including work past its stopping point)."""
    return {row["idx"]: row for rows in read_blocks(Path(run_dir)).values() for row in rows}


def _common(runs: list[dict[int, Row]]) -> list[int]:
    keys = set(runs[0])
    for run in runs[1:]:
        keys &= set(run)
    return sorted(keys)


def extract_failures(runs: list[dict[int, Row]]) -> list[int]:
    """Indices that at least one run did not solve correctly."""
    return sorted({idx for run in runs for idx, row in run.items() if row["status"] != "correct"})


def union(runs: list[dict[int, Row]]) -> dict[int, bool]:
    """Oracle union: solved if any run solved it correctly."""
    return {idx: any(run[idx]["status"] == "correct" for run in runs) for idx in _common(runs)}


def majority(runs: list[dict[int, Row]]) -> dict[int, str]:
    """Deployable vote without ground truth: the most frequent non-timeout
    answer, abstaining ("timeout") on ties or when every run timed out.
    Returns the status of the selected answer per idx."""
    out = {}
    for idx in _common(runs):
        answers = [run[idx] for run in runs if run[idx]["status"] != "timeout"]
        if not answers:
            out[idx] = "timeout"
            continue
        votes = Counter(row["solution"] for row in answers).most_common()
        if len(votes) > 1 and votes[0][1] == votes[1][1]:
            out[idx] = "timeout"
            continue
        winner = votes[0][0]
        out[idx] = next(row["status"] for row in answers if row["solution"] == winner)
    return out
