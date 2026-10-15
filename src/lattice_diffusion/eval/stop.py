"""Stopping rules for sequential evaluation.

A rule is checked after every puzzle of the contiguous evaluated prefix, in
index-list order. It only sees prefix counts, so where a run stops never
depends on how many workers ran it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Literal

from lattice_diffusion.eval.stats import confidence_sequence

Status = Literal["correct", "wrong", "timeout"]
StopReason = Literal["failures", "half_width", "fixed_n", "exhausted"]


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


@dataclass(frozen=True)
class NoStop:
    """Evaluate every index."""

    KIND: ClassVar[str] = "none"
    limit: ClassVar[int | None] = None

    def should_stop(self, counts: PrefixCounts) -> StopReason | None:
        return None

    def to_dict(self) -> dict:
        return {"kind": self.KIND, **asdict(self)}


@dataclass(frozen=True)
class SequentialStop:
    """Stop at `max_failures` failures or a CS half-width <= `half_width`, never before `n_min`.

    The half-width is checked every `check_every` puzzles.
    """

    KIND: ClassVar[str] = "sequential"
    limit: ClassVar[int | None] = None

    max_failures: int = 100
    half_width: float = 0.001
    n_min: int = 1000
    alpha: float = 0.05
    check_every: int = 100

    def should_stop(self, counts: PrefixCounts) -> StopReason | None:
        if counts.n < self.n_min:
            return None
        if counts.failures >= self.max_failures:
            return "failures"
        if counts.n % self.check_every == 0:
            lo, hi = confidence_sequence(counts.correct, counts.n, self.alpha)
            if (hi - lo) / 2 <= self.half_width:
                return "half_width"
        return None

    def to_dict(self) -> dict:
        return {"kind": self.KIND, **asdict(self)}


@dataclass(frozen=True)
class FixedN:
    """Evaluate exactly the first `n` indices, so every ablation arm sees the same puzzles."""

    KIND: ClassVar[str] = "fixed_n"

    n: int

    @property
    def limit(self) -> int:
        return self.n

    def should_stop(self, counts: PrefixCounts) -> StopReason | None:
        return "fixed_n" if counts.n >= self.n else None

    def to_dict(self) -> dict:
        return {"kind": self.KIND, **asdict(self)}


StopRule = NoStop | SequentialStop | FixedN

PRESETS: dict[str, StopRule] = {
    "sequential": SequentialStop(),
    "full": NoStop(),
    "ablation": FixedN(1000),
}

_RULES = {rule.KIND: rule for rule in (NoStop, SequentialStop, FixedN)}


def stop_from_dict(d: dict) -> StopRule:
    """Inverse of `StopRule.to_dict`."""
    d = dict(d)
    kind = d.pop("kind")
    if kind not in _RULES:
        raise ValueError(f"unknown stop rule kind {kind!r}")
    return _RULES[kind](**d)
