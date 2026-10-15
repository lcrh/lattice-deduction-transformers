"""Stopping rules for sequential evaluation.

A rule is consulted after every puzzle of the contiguous evaluated prefix
(in index-list order) and returns a stop reason or None. Rules only see
prefix counts, so where a run stops depends on the index list, the eval
seed and the block size, never on how many workers ran it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from lattice_diffusion.eval.stats import confidence_sequence


@dataclass
class PrefixCounts:
    n: int = 0
    correct: int = 0
    wrong: int = 0
    timeout: int = 0

    @property
    def failures(self) -> int:
        return self.wrong + self.timeout

    def add(self, status: str) -> None:
        self.n += 1
        if status == "correct":
            self.correct += 1
        elif status == "wrong":
            self.wrong += 1
        elif status == "timeout":
            self.timeout += 1
        else:
            raise ValueError(f"unknown status {status!r}")


@dataclass(frozen=True)
class NoStop:
    """Evaluate every index."""

    limit = None

    def should_stop(self, counts: PrefixCounts) -> str | None:
        return None

    def to_dict(self) -> dict:
        return {"kind": "none"}


@dataclass(frozen=True)
class SequentialStop:
    """Stop once the error rate is pinned down.

    Fires on the first of:
      - `failures`:   wrong + timeout >= max_failures;
      - `half_width`: the confidence-sequence half-width on accuracy is
                      <= half_width (checked every `check_every` puzzles);
    but never before `n_min` puzzles. Running out of indices is reported by
    the scheduler as `exhausted`.
    """

    max_failures: int = 100
    half_width: float = 0.001
    n_min: int = 1000
    alpha: float = 0.05
    check_every: int = 100

    limit = None

    def should_stop(self, counts: PrefixCounts) -> str | None:
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
        return {"kind": "sequential", **asdict(self)}


@dataclass(frozen=True)
class FixedN:
    """Evaluate exactly the first `n` indices (`fixed_n`), or the whole list
    if it is shorter (`exhausted`). Used for ablation tables, so every arm is
    scored on the same puzzles."""

    n: int

    @property
    def limit(self) -> int:
        return self.n

    def should_stop(self, counts: PrefixCounts) -> str | None:
        return "fixed_n" if counts.n >= self.n else None

    def to_dict(self) -> dict:
        return {"kind": "fixed_n", "n": self.n}


StopRule = NoStop | SequentialStop | FixedN

PRESETS: dict[str, StopRule] = {
    "sequential": SequentialStop(),
    "full": NoStop(),
    "ablation": FixedN(1000),
}


def stop_from_dict(d: dict) -> StopRule:
    d = dict(d)
    kind = d.pop("kind")
    if kind == "none":
        return NoStop()
    if kind == "sequential":
        return SequentialStop(**d)
    if kind == "fixed_n":
        return FixedN(**d)
    raise ValueError(f"unknown stop rule kind {kind!r}")
