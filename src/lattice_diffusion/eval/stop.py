"""Stopping rules for sequential evaluation.

A rule is checked after every puzzle of the contiguous evaluated prefix, in
index-list order. It only sees prefix counts, so where a run stops never
depends on how many workers ran it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar, Literal

from lattice_diffusion.eval.stats import confidence_sequence

Status = Literal["correct", "wrong", "timeout"]
StopReason = Literal["failures", "half_width", "n_max", "fixed_n", "exhausted"]


@dataclass
class PrefixCounts:
    """Outcome counts over the evaluated prefix."""

    n: int = 0
    correct: int = 0
    wrong: int = 0
    timeout: int = 0

    @property
    def failures(self) -> int:
        return self.wrong + self.timeout

    def add(self, status: Status) -> None:
        if status not in ("correct", "wrong", "timeout"):
            raise ValueError(f"unknown status {status!r}")
        self.n += 1
        setattr(self, status, getattr(self, status) + 1)


class StopRule(ABC):
    """Decides, from prefix counts, when an eval has seen enough puzzles."""

    KIND: ClassVar[str]
    limit: ClassVar[int | None] = None     # never needs items past this position

    @abstractmethod
    def should_stop(self, counts: PrefixCounts) -> StopReason | None: ...

    def to_dict(self) -> dict:
        return {"kind": self.KIND, **asdict(self)}


@dataclass(frozen=True)
class NoStop(StopRule):
    """Evaluate every index."""

    KIND: ClassVar[str] = "none"

    def should_stop(self, counts: PrefixCounts) -> StopReason | None:
        return None


@dataclass(frozen=True)
class SequentialStop(StopRule):
    """Stop once failures or the accuracy interval's width cross their thresholds.

    Fires at `max_failures` failures, or when the confidence-sequence half-width
    is <= `half_width` (checked every `check_every` puzzles, from `n_min` on).
    With `n_max`, also stops after that many puzzles.
    """

    KIND: ClassVar[str] = "sequential"

    max_failures: int = 100
    half_width: float = 0.001
    n_min: int = 1000
    alpha: float = 0.05
    check_every: int = 100
    n_max: int | None = None

    @property
    def limit(self) -> int | None:
        return self.n_max

    def should_stop(self, counts: PrefixCounts) -> StopReason | None:
        if self.n_max is not None and counts.n >= self.n_max:
            return "n_max"
        if counts.failures >= self.max_failures:
            return "failures"
        if counts.n >= self.n_min and counts.n % self.check_every == 0:
            lo, hi = confidence_sequence(counts.correct, counts.n, self.alpha)
            if (hi - lo) / 2 <= self.half_width:
                return "half_width"
        return None


@dataclass(frozen=True)
class FixedN(StopRule):
    """Evaluate exactly the first `n` indices, so every ablation arm sees the same puzzles."""

    KIND: ClassVar[str] = "fixed_n"

    n: int

    @property
    def limit(self) -> int:
        return self.n

    def should_stop(self, counts: PrefixCounts) -> StopReason | None:
        return "fixed_n" if counts.n >= self.n else None


PRESETS: dict[str, StopRule] = {
    "sequential": SequentialStop(),
    "full": NoStop(),
    # Appendix ablations: 8 blocks of 512; raise chosen entries to 20 blocks later (reuses the first 8).
    "ablation": SequentialStop(n_max=4096),
    "ablation_10k": SequentialStop(n_max=10240),
    "select": SequentialStop(n_max=4000),   # phase-1 default selection on sudoku_val
}

_RULES = {rule.KIND: rule for rule in (NoStop, SequentialStop, FixedN)}


def stop_from_dict(d: dict) -> StopRule:
    """Inverse of `StopRule.to_dict`."""
    d = dict(d)
    kind = d.pop("kind")
    if kind not in _RULES:
        raise ValueError(f"unknown stop rule kind {kind!r}")
    return _RULES[kind](**d)
