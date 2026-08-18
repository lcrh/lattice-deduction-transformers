"""Test-time symmetry-frame ensembling (CoLT H2, ad3002/colt
scripts/h2_symmetry_frames.py) for PowersetModel.

Wraps a trained model so that every forward pass runs K digit-permutation
frames (permute the channel axis of the lattice, forward, invert the
permutation on the logits) and aggregates:

  * ``union`` — elementwise max over frames of the BCE (elimination) logits:
                keep a candidate if ANY frame keeps it. Maximally conservative
                elimination — the direct antidote to value-specific first-pass
                poisoning (a candidate is only eliminated if every frame
                agrees to eliminate it).
  * ``mean``  — average logits across frames (plain ensemble; control).

The softmax (branch), conflict, and policy heads are always mean-aggregated
(union has no meaning for them). Frame 0 is the identity permutation, so
K=1 reproduces the bare model exactly.

No retraining required — this is a pure eval-time wrapper: pass the wrapped
model anywhere `solve()` / `dpll_step` expects a model. It composes cleanly
with `StepConfig.augment` (per-row aug in dpll_step): the wrapper's digit
perms just stack on top of whatever frame the state arrives in.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UnionFrames(nn.Module):
    """K digit-permutation frames per forward; aggregate inverse-mapped logits."""

    def __init__(self, model: nn.Module, n_frames: int, agg: str = "union",
                 generator: torch.Generator | None = None):
        super().__init__()
        if agg not in ("union", "mean"):
            raise ValueError(f"unknown agg {agg!r}")
        self.model = model
        self.n_frames = n_frames
        self.agg = agg
        self.gen = generator  # CPU generator; perms are moved to x.device

    def forward(self, x: torch.Tensor, return_all: bool = False,
                use_final: bool = False) -> dict[str, torch.Tensor]:
        if return_all:
            raise NotImplementedError(
                "UnionFrames is an eval-time wrapper; return_all (training) "
                "is not supported")
        B, S, C = x.shape
        frames: list[dict[str, torch.Tensor]] = []
        for k in range(self.n_frames):
            if k == 0:
                perm = torch.arange(C, device=x.device)   # identity frame first
            else:
                perm = torch.randperm(C, generator=self.gen).to(x.device)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(C, device=x.device)
            out = self.model(x[:, :, perm], use_final=use_final)
            f = {
                "bce": out["bce"][:, :, inv],             # back to original labels
                "softmax": out["softmax"][:, :, inv],
            }
            for key in ("cls", "conflict", "policy"):
                if key in out:
                    f[key] = out[key]
            frames.append(f)

        result: dict[str, torch.Tensor] = {}
        bce = torch.stack([f["bce"] for f in frames])
        result["bce"] = bce.amax(dim=0) if self.agg == "union" else bce.mean(dim=0)
        result["softmax"] = torch.stack([f["softmax"] for f in frames]).mean(dim=0)
        for key in ("cls", "conflict", "policy"):
            if key in frames[0]:
                result[key] = torch.stack([f[key] for f in frames]).mean(dim=0)
        return result
