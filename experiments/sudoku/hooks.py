"""Neutral extension seams for the shared DPLL / solve / train kernels.

Callers inject concrete strategies; core defaults to None and keeps the
legacy single-pass / uniform-decide / root-reset / discard-on-conflict path.
These objects are not serialized into eval JSON (see `public_asdict`).
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Protocol, runtime_checkable


def public_asdict(obj: Any) -> dict:
    """`asdict` that drops private/hook fields (name starts with `_` or
    has metadata `public=False`)."""
    if not is_dataclass(obj):
        raise TypeError(f"public_asdict expected a dataclass, got {type(obj)}")
    out = {}
    for f in fields(obj):
        if f.name.startswith("_"):
            continue
        if f.metadata.get("public") is False:
            continue
        val = getattr(obj, f.name)
        if is_dataclass(val):
            out[f.name] = public_asdict(val)
        else:
            out[f.name] = val
    return out


@runtime_checkable
class DeductionOperator(Protocol):
    """Optional multi-pass deduction controller inside `dpll_step`.

    When None, core runs exactly one forward+threshold pass.
    """

    def max_passes(self) -> int:
        """Upper bound on the deduce loop (including fixpoint caps)."""

    def should_stop(
        self,
        *,
        pass_index: int,
        max_passes: int,
        nothing_eliminated: bool,
        all_done: bool,
    ) -> bool:
        """Return True to leave the deduce loop after the just-finished pass."""


@runtime_checkable
class DecisionPolicy(Protocol):
    """Optional cell/digit selection inside `dpll_step`.

    When None, core uses uniform multi-alive cell + softmax/argmax digit.
    """

    def pick(
        self,
        *,
        cfg,
        new_state_aug,          # [B,S,C] mutated in place when pinning
        sm_logits,              # [B,S,C]
        can_decide,             # [B] bool
        vd: int,
        ip_aug,                 # [B,S] bool | None
        decide_rank,            # [B] long | None
        device,
    ) -> int:
        """Pin decided cells; return # of deciding rows (for stats)."""


@runtime_checkable
class SearchStrategy(Protocol):
    """Optional search / backtrack lifecycle for `solve()`."""

    def needs_snapshots(self) -> bool: ...

    def decide_rank(self, B: int, K: int, device):
        """Optional [B] long rank tensor for digit selection; else None."""

    def on_fill(self, rows: slice, *, snap) -> None: ...

    def after_step(self, *, snap, state, new_state, conflict, just_solved,
                   chain_done, info, vd: int) -> None: ...

    def resolve_conflicts(
        self,
        *,
        snap,
        idx,                    # [n] global chain rows in conflict
        local_reset,            # [n] slot-local indices
        new_state,
        original,
        slot_gt_idx,            # [S] GT digits for this slot
        vd: int,
        device,
    ) -> object:
        """Apply resets; return slot-local indices that landed at root
        (for sequential-attempt bookkeeping). Legacy: all local_reset."""

    def accumulate_step_diagnostics(
        self,
        *,
        info,
        active_rows,
        state,
        gt_one_hot,
        acc: dict,
    ) -> None: ...

    def result_extras(self) -> dict:
        """Extra kwargs for SolveResult (diagnostics)."""


@runtime_checkable
class PoolStrategy(Protocol):
    """Optional training-pool conflict handling for `train()`."""

    def attach(self, pool_size: int, S: int, C: int, device) -> object:
        """Allocate private state (or return None)."""

    def after_step(
        self,
        *,
        pool,
        sample_idx,
        state,
        new_state,
        detected_conflict,
        true_positive_conflict,
        solved,
        info,
        vd: int,
        age_exceeded,
    ) -> tuple[object, object, object]:
        """Return (new_state, restored_mask[B], discard_mask[B])."""

    def on_backfill(self, *, pool, sample_idx, discard) -> None: ...

    def log_extra(self, *, pool) -> str:
        """Optional one-line health string; empty to skip."""
