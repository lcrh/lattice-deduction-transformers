"""Eval run specs, on-disk layout, and summaries.

Layout under the eval root (on the checkpoint volume):

    <root>/<checkpoint stem>/<tag>-<spec hash>/
        spec.json                       hashed EvalSpec fields + launch provenance
        indices.sha256                  digest of the index list the blocks index into
        blocks/000000.jsonl             one `Row` per puzzle, in index-list order
        summary-<kind>-<digest8>.json   one per stop rule the blocks were summarized with

Block contents do not depend on the stop rule, so it is not hashed: a looser
or tighter rule reuses and extends the same blocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from ldt.config import stable_hash
from ldt.eval.scheduler import Row, ScheduleResult
from ldt.eval.stats import (
    binomial_upper, confidence_sequence, inverse_binomial_rate,
)


@dataclass
class EvalSpec:
    """Everything that determines an eval's results.

    Fields in `_UNHASHED` never change a block's rows: the stop rule only
    decides how many blocks are summarized, the rest affect cost or provenance.
    """

    task: Literal["sudoku", "snowflake", "maze"]
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
    # Task-specific restriction of the list, kept in list order (EvalTask.filter_indices),
    # e.g. {"orders": [7, 8, 9, 10]} for Snowflake.
    index_filter: dict | None = None
    n_workers: int = 1
    max_in_flight: int | None = None    # None = n_workers
    keep_warm_s: int = 10               # idle seconds before a worker scales down
    # Provenance (ldt.provenance): recorded, never hashed.
    exp_id: str = ""
    entry: str = ""
    code_sha: str = ""

    _UNHASHED: ClassVar[tuple[str, ...]] = ("stop", "n_workers", "max_in_flight", "keep_warm_s",
                 "exp_id", "entry", "code_sha")

    def hashed_fields(self) -> dict:
        d = asdict(self)
        for k in self._UNHASHED:
            d.pop(k)
        return d

    def spec_hash(self) -> str:
        return stable_hash(self.hashed_fields())

    @property
    def provenance(self) -> dict:
        return {"exp_id": self.exp_id, "entry": self.entry, "code_sha": self.code_sha}

    @property
    def window(self) -> int:
        return self.max_in_flight if self.max_in_flight is not None else self.n_workers

    def run_dir(self, root: str | Path) -> Path:
        return Path(root) / Path(self.checkpoint).stem / f"{self.tag}-{self.spec_hash()}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> EvalSpec:
        return cls(**json.loads(s))


def block_path(run_dir: Path, block: int) -> Path:
    return run_dir / "blocks" / f"{block:06d}.jsonl"


def write_block(run_dir: Path, block: int, rows: list[Row], meta: dict | None = None) -> None:
    """Atomic write so a killed coordinator never leaves a partial block.

    `meta` (worker stats such as model calls and seconds) goes to a sidecar
    written first, so a block file on disk always has its stats when they exist.
    """
    path = block_path(run_dir, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta is not None:
        meta_path = path.with_suffix(".meta.json")
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, meta_path)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def read_block_meta(run_dir: Path) -> dict[int, dict]:
    return {int(p.name.split(".")[0]): json.loads(p.read_text())
            for p in (run_dir / "blocks").glob("*.meta.json")}


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
    if path.exists():
        if json.loads(path.read_text())["hashed"] != spec.hashed_fields():
            raise RuntimeError(f"{path} exists with a different spec")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hashed": spec.hashed_fields(), "provenance": spec.provenance},
                               sort_keys=True, indent=2))


def check_or_write_digest(run_dir: Path, digest: str, name: str) -> None:
    """Record which index list a run's blocks index into; refuse a different one on resume."""
    path = run_dir / "indices.sha256"
    if path.exists():
        if path.read_text().strip() != digest:
            raise RuntimeError(f"{run_dir} was evaluated on a different {name} list")
        return
    path.write_text(digest + "\n")


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
        **{f"p{q}": _percentile(vals, q / 100) for q in (50, 75, 90, 95, 99)},
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
    calls_per_solve: float | None       # batched model calls / correct, over the reported blocks
    n_evaluated: int                    # including work past the stopping point
    overshoot: int
    wall_seconds: float
    gpu_seconds: float                  # worker time spent solving every evaluated block
    startup_seconds: float              # worker time loading/compiling the model, every block
    model_calls_total: int | None       # batched forwards over every evaluated block
    mean_rounds_correct: float | None   # rounds to accept, over correct puzzles
    mean_resets: float | None
    alpha: float = 0.05
    soundness: dict = field(default_factory=dict)   # pooled over the reported prefix
    labels: dict = field(default_factory=dict)      # outcome label counts (Maze)


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _calls_per_solve(result: ScheduleResult, meta: dict[int, dict]) -> float | None:
    """Batched model calls per correct puzzle over the blocks the reported prefix touches.

    Model calls are only known per block, so the last block counts in full,
    including any puzzles past the stopping point.
    """
    blocks = sorted({r["block"] for r in result.rows})
    if not blocks or any(b not in meta for b in blocks):
        return None
    calls = sum(meta[b]["model_calls"] for b in blocks)
    correct = sum(r["status"] == "correct" for b in blocks for r in result.blocks[b])
    return calls / correct if correct else None


def _failure_rate_inverse_binomial(spec: EvalSpec, c, stop_reason: str) -> float:
    """(K - 1) / (n - 1) when the run stopped at its K-th failure, else failures / n."""
    if stop_reason == "failures" and c.failures == spec.stop.get("max_failures"):
        return inverse_binomial_rate(c.failures, c.n)
    return c.failures / c.n if c.n else 0.0


def summarize(spec: EvalSpec, run_dir: Path, result: ScheduleResult,
              wall_seconds: float, alpha: float = 0.05) -> Summary:
    c = result.counts
    meta = read_block_meta(run_dir)
    labels = Counter(r["label"] for r in result.rows if "label" in r)
    return Summary(
        spec=asdict(spec),
        run_dir=str(run_dir),
        n=c.n,
        stop_reason=result.stop_reason,
        correct=c.correct,
        wrong=c.wrong,
        timeout=c.timeout,
        accuracy=c.correct / c.n if c.n else 0.0,
        accuracy_cs=confidence_sequence(c.correct, c.n, alpha),
        failure_rate=c.failures / c.n if c.n else 0.0,
        failure_rate_inverse_binomial=_failure_rate_inverse_binomial(spec, c, result.stop_reason),
        wrong_rate_upper=binomial_upper(c.wrong, c.n, alpha),
        calls_all=_call_stats(result.rows),
        calls_correct=_call_stats([r for r in result.rows if r["status"] == "correct"]),
        calls_per_solve=_calls_per_solve(result, meta),
        n_evaluated=result.n_evaluated,
        overshoot=result.n_evaluated - c.n,
        wall_seconds=wall_seconds,
        gpu_seconds=sum(m["seconds"] for m in meta.values()),
        startup_seconds=sum(m["startup_seconds"] for m in meta.values()),
        model_calls_total=sum(m["model_calls"] for m in meta.values()) if meta else None,
        mean_rounds_correct=_mean([r["rounds"] for r in result.rows if r["status"] == "correct"]),
        mean_resets=_mean([r["n_resets"] for r in result.rows]),
        alpha=alpha,
        soundness=_soundness(result.rows),
        labels=dict(sorted(labels.items())),
    )


def summary_path(run_dir: Path, stop: dict) -> Path:
    digest = hashlib.sha256(json.dumps(stop, sort_keys=True).encode()).hexdigest()[:8]
    return run_dir / f"summary-{stop['kind']}-{digest}.json"


def write_summary(run_dir: Path, summary: Summary) -> None:
    path = summary_path(run_dir, summary.spec["stop"])
    path.write_text(json.dumps(asdict(summary), indent=2))


def _fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def format_summary(s: Summary) -> str:
    lo, hi = s.accuracy_cs
    lines = [
        f"n={s.n} ({s.stop_reason})  acc={s.accuracy:.5%} [{lo:.5%}, {hi:.5%}]  "
        f"correct={s.correct} wrong={s.wrong} (rate <= {s.wrong_rate_upper:.2e}) "
        f"timeout={s.timeout}  overshoot={s.overshoot}",
        f"model calls={_fmt(s.model_calls_total, ',')} "
        f"(per solve {_fmt(s.calls_per_solve, ',.0f')})  "
        f"rounds (correct)={_fmt(s.mean_rounds_correct, '.1f')}  "
        f"resets={_fmt(s.mean_resets, '.2f')}  "
        f"wall={s.wall_seconds:.0f}s gpu={s.gpu_seconds / 3600:.2f}h "
        f"startup={s.startup_seconds / 3600:.2f}h",
    ]
    if s.soundness:
        sd = s.soundness
        lines.append(f"unsound eliminations={sd['unsound']}/{sd['deduced']} "
                     f"({sd['unsound_rate']:.4%})  conflict head "
                     f"P={_fmt(sd['cls_precision'], '.3f')} R={_fmt(sd['cls_recall'], '.3f')}")
    if s.labels:
        lines.append("outcomes: " + "  ".join(f"{k}={v}" for k, v in s.labels.items()))
    return "\n".join(lines)
