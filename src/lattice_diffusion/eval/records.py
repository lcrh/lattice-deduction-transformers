"""Eval run specs, on-disk layout, and summaries.

Layout under the eval root (on the checkpoint volume):

    <root>/<checkpoint stem>/<tag>-<spec hash>/
        spec.json            the resolved EvalSpec (hashed fields only)
        blocks/000000.jsonl  one row per puzzle, in index-list order
        summary-<rule>.json  one per stop rule the blocks were summarized with

Block contents do not depend on the stop rule, so the rule is not part of
the hash: rerunning with a looser or tighter rule (or `NoStop`) reuses and
extends the same blocks.

Row fields (every benchmark):
    idx       absolute dataset index
    pos       position in the index list
    block     block id
    status    "correct" | "wrong" | "timeout"
    rounds    solver round at which the puzzle was accepted (-1 if never)
    calls     model calls while the puzzle occupied a solver slot
    forwards  single-chain forwards if run unbatched
    n_resets  chain resets
    solution  the accepted prediction, benchmark-specific string encoding
    deduced, unsound, cls
              soundness diagnostics: candidates eliminated over the
              puzzle's chain-rounds, how many were the solution's own, and
              the conflict head's [tp, fp, fn, tn] against true conflicts
    verified  rule check of `solution` against the puzzle (never used for
              acceptance; recorded for portfolio selection and auditing);
              null where the benchmark has no rule check (Snowflake)
Optional fields:
    forwards_seq  forwards if one chain ran sequentially (drained-chain
                  estimate; only with `solve.estimate_sequential`)
    label         Maze only: the exact-vs-alternate-path outcome
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lattice_diffusion.eval.scheduler import Row, ScheduleResult
from lattice_diffusion.eval.stats import (
    binomial_upper, confidence_sequence, inverse_binomial_rate,
)


@dataclass
class EvalSpec:
    """Everything that determines an eval's results. Fields in `_UNHASHED`
    do not change any block's rows: the stop rule only decides how many
    blocks are summarized, the rest only affect cost and wall-clock time."""

    task: str                           # "sudoku" | "snowflake" | "maze"
    checkpoint: str
    index_list: str                     # named list, or "explicit"
    solve: dict                         # task-specific solver config
    stop: dict                          # StopRule.to_dict()
    block_size: int = 512
    eval_seed: int = 0
    compile: bool = False
    tag: str = "test"
    explicit_indices: list[int] | None = None
    explicit_split: str | None = None
    # Task-specific restriction of the index list, applied in list order,
    # e.g. {"orders": [7, 8, 9, 10]} for Snowflake (EvalTask.filter_indices).
    index_filter: dict | None = None
    n_workers: int = 32
    max_in_flight: int | None = None    # None = n_workers
    keep_warm_s: int = 300              # idle seconds before a worker scales down

    _UNHASHED = ("stop", "n_workers", "max_in_flight", "keep_warm_s")

    def hashed_fields(self) -> dict:
        d = asdict(self)
        for k in self._UNHASHED:
            d.pop(k)
        return d

    def spec_hash(self) -> str:
        blob = json.dumps(self.hashed_fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def window(self) -> int:
        return self.max_in_flight if self.max_in_flight is not None else self.n_workers

    def run_dir(self, root: str | Path) -> Path:
        return Path(root) / Path(self.checkpoint).stem / f"{self.tag}-{self.spec_hash()}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "EvalSpec":
        return cls(**json.loads(s))


def block_path(run_dir: Path, block: int) -> Path:
    return run_dir / "blocks" / f"{block:06d}.jsonl"


def write_block(run_dir: Path, block: int, rows: list[Row]) -> None:
    """Atomic write so a killed coordinator never leaves a partial block."""
    path = block_path(run_dir, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def read_blocks(run_dir: Path) -> dict[int, list[Row]]:
    blocks_dir = run_dir / "blocks"
    if not blocks_dir.exists():
        return {}
    out = {}
    for path in sorted(blocks_dir.glob("*.jsonl")):
        with path.open() as fh:
            out[int(path.stem)] = [json.loads(line) for line in fh if line.strip()]
    return out


def check_or_write_spec(run_dir: Path, spec: EvalSpec) -> None:
    """Refuse to mix blocks from different specs in one run directory."""
    path = run_dir / "spec.json"
    hashed = json.dumps(spec.hashed_fields(), sort_keys=True, indent=2)
    if path.exists():
        if path.read_text() != hashed:
            raise RuntimeError(f"{path} exists with a different spec")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(hashed)


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _call_stats(rows: list[Row]) -> dict:
    vals = sorted(r["calls"] for r in rows)
    return {
        "mean": sum(vals) / len(vals) if vals else None,
        **{f"p{int(q * 100)}": _percentile(vals, q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
    }


def _soundness(rows: list[Row]) -> dict:
    """Pooled soundness diagnostics, when the rows carry them."""
    if not rows or "deduced" not in rows[0]:
        return {}
    deduced = sum(r["deduced"] for r in rows)
    unsound = sum(r["unsound"] for r in rows)
    tp, fp, fn, tn = (sum(r["cls"][i] for r in rows) for i in range(4))
    out = {
        "deduced": deduced, "unsound": unsound,
        "unsound_rate": unsound / deduced if deduced else 0.0,
        "cls_precision": tp / (tp + fp) if tp + fp else None,
        "cls_recall": tp / (tp + fn) if tp + fn else None,
        "cls_counts": [tp, fp, fn, tn],
    }
    if "pass_deduced_unsound" in rows[0]:
        n_pass = len(rows[0]["pass_deduced_unsound"])
        per_pass = [[sum(r["pass_deduced_unsound"][i][k] for r in rows) for k in (0, 1)]
                    for i in range(n_pass)]
        out["unsound_rate_per_pass"] = [u / d if d else None for d, u in per_pass]
    return out


@dataclass
class Summary:
    spec: dict
    run_dir: str
    n: int
    stop_reason: str
    correct: int
    wrong: int
    timeout: int
    accuracy: float
    accuracy_cs: tuple[float, float]
    failure_rate: float                 # (wrong + timeout) / n
    failure_rate_inverse_binomial: float
    wrong_rate_upper: float             # one-sided 95% Clopper-Pearson
    calls_all: dict
    calls_correct: dict
    n_evaluated: int                    # including work past the stopping point
    overshoot: int
    wall_seconds: float
    gpu_seconds: float                  # worker time spent solving blocks
    startup_seconds: float              # worker time loading/compiling the model
    alpha: float = 0.05
    soundness: dict = field(default_factory=dict)   # pooled over the reported prefix
    extra: dict = field(default_factory=dict)


def summarize(spec: EvalSpec, run_dir: Path, result: ScheduleResult,
              wall_seconds: float, gpu_seconds: float, startup_seconds: float = 0.0,
              alpha: float = 0.05) -> Summary:
    c = result.counts
    n = c.n
    return Summary(
        spec=asdict(spec),
        run_dir=str(run_dir),
        n=n,
        stop_reason=result.stop_reason,
        correct=c.correct,
        wrong=c.wrong,
        timeout=c.timeout,
        accuracy=c.correct / n if n else 0.0,
        accuracy_cs=confidence_sequence(c.correct, n, alpha),
        failure_rate=c.failures / n if n else 0.0,
        failure_rate_inverse_binomial=inverse_binomial_rate(c.failures, n),
        wrong_rate_upper=binomial_upper(c.wrong, n, alpha),
        calls_all=_call_stats(result.rows),
        calls_correct=_call_stats([r for r in result.rows if r["status"] == "correct"]),
        n_evaluated=result.n_evaluated,
        overshoot=result.n_evaluated - n,
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
        startup_seconds=startup_seconds,
        alpha=alpha,
        soundness=_soundness(result.rows),
    )


def summary_path(run_dir: Path, stop: dict) -> Path:
    digest = hashlib.sha256(json.dumps(stop, sort_keys=True).encode()).hexdigest()[:8]
    return run_dir / f"summary-{stop['kind']}-{digest}.json"


def write_summary(run_dir: Path, summary: Summary) -> None:
    path = summary_path(run_dir, summary.spec["stop"])
    path.write_text(json.dumps(asdict(summary), indent=2))


def format_summary(s: Summary) -> str:
    lo, hi = s.accuracy_cs
    return (
        f"n={s.n} ({s.stop_reason})  acc={s.accuracy:.5%} [{lo:.5%}, {hi:.5%}]  "
        f"correct={s.correct} wrong={s.wrong} (rate <= {s.wrong_rate_upper:.2e}) "
        f"timeout={s.timeout}  overshoot={s.overshoot}  "
        f"wall={s.wall_seconds:.0f}s gpu={s.gpu_seconds / 3600:.2f}h "
        f"startup={s.startup_seconds / 3600:.2f}h"
    )
