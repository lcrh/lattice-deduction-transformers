"""Training-pool conflict strategy for matched-policy followup runs."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PoolOpts:
    backtrack: str = "root"
    geometric_p: float = 0.5
    learn_negation: bool = False
    snapshot_max_depth: int = 64


@dataclass
class SnapshotPoolStrategy:
    opts: PoolOpts = field(default_factory=PoolOpts)

    def attach(self, pool_size: int, S: int, C: int, device):
        if self.opts.backtrack == "root":
            return None
        D = self.opts.snapshot_max_depth
        return {
            "snap_state": torch.zeros(pool_size, D, S, C, dtype=torch.uint8, device=device),
            "snap_cell": torch.zeros(pool_size, D, dtype=torch.long, device=device),
            "snap_digit": torch.zeros(pool_size, D, dtype=torch.long, device=device),
            "depth": torch.zeros(pool_size, dtype=torch.long, device=device),
            "D_max": D,
            "negate_on": (self.opts.backtrack == "last+negate") or self.opts.learn_negation,
            "n_restores": 0,
        }

    def after_step(self, *, pool, sample_idx, state, new_state, detected_conflict,
                   true_positive_conflict, solved, info, vd: int, age_exceeded):
        if pool is None:
            restored = torch.zeros_like(solved)
            discard = solved | true_positive_conflict | age_exceeded
            return new_state, restored, discard

        # Snapshot push for non-conflict, non-solved rows that decided.
        deduce_mask = info["deduce_mask"]
        state_post = state.masked_fill(deduce_mask, 0.0)
        pre_alive = (state_post[..., :vd] > 0.5).sum(dim=-1)
        post_alive = (new_state[..., :vd] > 0.5).sum(dim=-1)
        decided = (pre_alive > 1) & (post_alive == 1)
        made = decided.any(dim=-1) & ~detected_conflict & ~solved
        depth = pool["depth"][sample_idx]
        D_max = pool["D_max"]
        has_room = depth < D_max
        push = made & has_room
        if push.any():
            rows = push.nonzero(as_tuple=True)[0]
            pidx = sample_idx[rows]
            cell_of = decided[rows].float().argmax(dim=-1)
            d_of = depth[rows]
            pool["snap_state"][pidx, d_of] = (state_post[rows] > 0.5).to(torch.uint8)
            pool["snap_cell"][pidx, d_of] = cell_of
            digit_of = new_state[rows, cell_of, :vd].argmax(dim=-1)
            pool["snap_digit"][pidx, d_of] = digit_of
            depth[rows] = d_of + 1
            pool["depth"][sample_idx] = depth
        overflow = made & ~has_room
        if overflow.any():
            depth[overflow] = depth[overflow] + 1
            pool["depth"][sample_idx] = depth

        # Restore on true-positive conflict.
        restored = torch.zeros_like(solved)
        tp = true_positive_conflict
        if tp.any():
            rows = tp.nonzero(as_tuple=True)[0]
            pidx = sample_idx[rows]
            d_conf = pool["depth"][pidx]
            bt = self.opts.backtrack
            if bt in ("last", "last+negate"):
                keep = (d_conf - 1).clamp(min=0)
            elif bt == "geometric":
                j = torch.empty(rows.numel(), device=state.device).geometric_(
                    self.opts.geometric_p).long().clamp(min=1)
                keep = (d_conf - j).clamp(min=0)
            elif bt == "uniform_depth":
                r = torch.rand(rows.numel(), device=state.device)
                keep = (r * d_conf.float()).floor().long().clamp(min=0)
            else:
                keep = torch.zeros_like(d_conf)
            force_root = (d_conf <= 0) | (d_conf > D_max)
            for jj in range(rows.numel()):
                if bool(force_root[jj].item()):
                    continue
                b = int(rows[jj].item())
                p = int(pidx[jj].item())
                k = int(keep[jj].item())
                new_state[b] = pool["snap_state"][p, k].float()
                if pool["negate_on"]:
                    nc = int(pool["snap_cell"][p, k].item())
                    nd = int(pool["snap_digit"][p, k].item())
                    new_state[b, nc, nd] = 0.0
                    if float(new_state[b, nc, :vd].sum().item()) < 0.5:
                        continue  # empty -> discard via not marking restored
                pool["depth"][p] = k
                restored[b] = True
                pool["n_restores"] += 1

        discard = solved | (true_positive_conflict & ~restored) | age_exceeded
        return new_state, restored, discard

    def on_backfill(self, *, pool, sample_idx, discard) -> None:
        if pool is None or not discard.any():
            return
        pool["depth"][sample_idx[discard]] = 0

    def log_extra(self, *, pool) -> str:
        if pool is None:
            return ""
        dep = pool["depth"].float()
        return (f"pool-bt={self.opts.backtrack} depth_p50={int(pool['depth'].median().item())} "
                f"depth_mean={float(dep.mean().item()):.1f} "
                f"depth_max={int(pool['depth'].max().item())} "
                f"restores={pool['n_restores']}")


def make_pool_strategy(
    *,
    backtrack: str = "root",
    geometric_p: float = 0.5,
    learn_negation: bool = False,
    snapshot_max_depth: int = 64,
) -> SnapshotPoolStrategy | None:
    if backtrack == "root" and not learn_negation:
        return None
    return SnapshotPoolStrategy(PoolOpts(
        backtrack=backtrack, geometric_p=geometric_p,
        learn_negation=learn_negation,
        snapshot_max_depth=snapshot_max_depth,
    ))
