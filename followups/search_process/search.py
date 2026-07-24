"""Search / backtrack strategies for followup eval and matched training.

Core `solve()` keeps root-reset behavior when `SolveConfig._search` is None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SearchOpts:
    backtrack: str = "root"          # root|last|geometric|uniform_depth|last+negate
    geometric_p: float = 0.5
    learn_negation: bool = False
    snapshot_max_depth: int = 64
    digit_policy: str = "softmax"    # for decide_rank allocation


@dataclass
class SnapshotSearchStrategy:
    """Partial / negating backtracking with optional decision-rank support."""
    opts: SearchOpts = field(default_factory=SearchOpts)
    conflict_depths: list[int] = field(default_factory=list)
    backtrack_targets: list[int] = field(default_factory=list)
    n_negations: int = 0
    n_unsound_negations: int = 0
    per_pass_deduced_total: list[int] = field(default_factory=list)
    per_pass_unsound_total: list[int] = field(default_factory=list)

    def needs_snapshots(self) -> bool:
        return self.opts.backtrack != "root"

    def decide_rank(self, B: int, K: int, device):
        if self.opts.digit_policy == "rank_k":
            return torch.arange(B, device=device) % K
        return None

    def attach(self, B: int, S: int, C: int, device) -> dict | None:
        if not self.needs_snapshots():
            return None
        D = self.opts.snapshot_max_depth
        return {
            "snap_state": torch.zeros(B, D, S, C, dtype=torch.uint8, device=device),
            "snap_cell": torch.zeros(B, D, dtype=torch.long, device=device),
            "snap_digit": torch.zeros(B, D, dtype=torch.long, device=device),
            "chain_depth": torch.zeros(B, dtype=torch.long, device=device),
            "D_max": D,
            "negate_on": (self.opts.backtrack == "last+negate") or self.opts.learn_negation,
        }

    def on_fill(self, rows: slice, *, snap) -> None:
        if snap is not None:
            snap["chain_depth"][rows] = 0

    def after_step(self, *, snap, state, new_state, conflict, just_solved,
                   chain_done, info, vd: int) -> None:
        if snap is None:
            return
        deduce_mask_s = info["deduce_mask"]
        state_post_deduce = state.masked_fill(deduce_mask_s, 0.0)
        pre_alive = (state_post_deduce[..., :vd] > 0.5).sum(dim=-1)
        post_alive = (new_state[..., :vd] > 0.5).sum(dim=-1)
        decided_cell_mask = (pre_alive > 1) & (post_alive == 1)
        made_decision = (
            decided_cell_mask.any(dim=-1) & ~conflict
            & ~just_solved & ~chain_done
        )
        chain_depth = snap["chain_depth"]
        D_max = snap["D_max"]
        has_room = chain_depth < D_max
        push_rows = (made_decision & has_room).nonzero(as_tuple=True)[0]
        if push_rows.numel() > 0:
            cell_of = decided_cell_mask[push_rows].float().argmax(dim=-1)
            depth_of = chain_depth[push_rows]
            snap["snap_state"][push_rows, depth_of] = (
                state_post_deduce[push_rows] > 0.5
            ).to(torch.uint8)
            snap["snap_cell"][push_rows, depth_of] = cell_of
            digit_of = new_state[push_rows, cell_of, :vd].argmax(dim=-1)
            snap["snap_digit"][push_rows, depth_of] = digit_of
            chain_depth[push_rows] = depth_of + 1
        overflow_rows = (made_decision & ~has_room)
        if overflow_rows.any():
            chain_depth[overflow_rows] = chain_depth[overflow_rows] + 1

    def resolve_conflicts(
        self, *, snap, idx, local_reset, new_state, original,
        slot_gt_idx, vd: int, device,
    ):
        """Apply policy resets. Return slot-local rows that landed at root."""
        if snap is None:
            new_state[idx] = original[idx]
            return local_reset

        n_conf = int(local_reset.numel())
        depth_at_conf = snap["chain_depth"][idx]
        bt = self.opts.backtrack
        if bt in ("last", "last+negate"):
            keep = (depth_at_conf - 1).clamp(min=0)
        elif bt == "geometric":
            j = torch.empty(n_conf, device=device).geometric_(
                self.opts.geometric_p).long().clamp(min=1)
            keep = (depth_at_conf - j).clamp(min=0)
        elif bt == "uniform_depth":
            r = torch.rand(n_conf, device=device)
            keep = (r * depth_at_conf.float()).floor().long().clamp(min=0)
        else:
            raise ValueError(f"unknown backtrack {bt!r}")

        D_max = snap["D_max"]
        force_root = (depth_at_conf > D_max) | (depth_at_conf <= 0)
        negate_on = snap["negate_on"]
        is_root = torch.zeros(n_conf, dtype=torch.bool, device=device)
        for jj in range(n_conf):
            b = int(idx[jj].item())
            d_conf = int(depth_at_conf[jj].item())
            if bool(force_root[jj].item()):
                new_state[b] = original[b]
                snap["chain_depth"][b] = 0
                is_root[jj] = True
                self.conflict_depths.append(d_conf)
                self.backtrack_targets.append(0)
                continue
            k = int(keep[jj].item())
            new_state[b] = snap["snap_state"][b, k].float()
            landed_root = (k == 0) and not negate_on
            if negate_on:
                nc = int(snap["snap_cell"][b, k].item())
                nd = int(snap["snap_digit"][b, k].item())
                gt_here = int(slot_gt_idx[nc].item())
                self.n_negations += 1
                if nd == gt_here:
                    self.n_unsound_negations += 1
                new_state[b, nc, nd] = 0.0
                if float(new_state[b, nc, :vd].sum().item()) < 0.5:
                    new_state[b] = original[b]
                    k = 0
                    landed_root = True
            snap["chain_depth"][b] = k
            is_root[jj] = landed_root
            self.conflict_depths.append(d_conf)
            self.backtrack_targets.append(k)
        return local_reset[is_root]

    def accumulate_step_diagnostics(self, *, info, active_rows, state, gt_one_hot, acc):
        per_pass_masks = info.get("per_pass_deduce_masks") or []
        for pass_i, pass_mask in enumerate(per_pass_masks):
            d = int((pass_mask.sum(dim=(1, 2)) * active_rows).sum().item())
            u = int(((pass_mask & ((state > 0.5) & gt_one_hot)).sum(dim=(1, 2))
                     * active_rows).sum().item())
            while pass_i >= len(self.per_pass_deduced_total):
                self.per_pass_deduced_total.append(0)
                self.per_pass_unsound_total.append(0)
            self.per_pass_deduced_total[pass_i] += d
            self.per_pass_unsound_total[pass_i] += u

    def result_extras(self) -> dict:
        return {
            "per_pass_deduced_total": list(self.per_pass_deduced_total),
            "per_pass_unsound_total": list(self.per_pass_unsound_total),
            "backtrack_policy": self.opts.backtrack,
            "conflict_depths": list(self.conflict_depths),
            "backtrack_targets": list(self.backtrack_targets),
            "n_negations": self.n_negations,
            "n_unsound_negations": self.n_unsound_negations,
        }


def make_search_strategy(
    *,
    backtrack: str = "root",
    geometric_p: float = 0.5,
    learn_negation: bool = False,
    snapshot_max_depth: int = 64,
    digit_policy: str = "softmax",
    track_per_pass: bool = False,
) -> SnapshotSearchStrategy | None:
    """None when fully legacy (root + no rank_k + no per-pass tracking)."""
    if (backtrack == "root" and digit_policy != "rank_k"
            and not learn_negation and not track_per_pass):
        return None
    return SnapshotSearchStrategy(SearchOpts(
        backtrack=backtrack, geometric_p=geometric_p,
        learn_negation=learn_negation,
        snapshot_max_depth=snapshot_max_depth,
        digit_policy=digit_policy,
    ))
