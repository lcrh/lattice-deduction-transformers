"""Per-row digit / dihedral augmentation in batched torch.

Each chain gets a different but equivalent view of its puzzle: digit labels
are permuted, then the n x n grid is rotated / reflected (D4). Conventions:
`digit_perm[b, c_new] = c_old` and `cell_perms[g, s_new] = s_old`, both applied
with `gather`; `invert_aug_state` undoes them in reverse order.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from experiments.sudoku.dpll import StepConfig


def build_dihedral_cell_perms(n: int = 9, device: torch.device | None = None) -> torch.Tensor:
    """[8, n*n] cell permutations of the D4 elements, indexed `k_rot * 2 + reflect`.

    The reflection is the dataset's (np.flip on axis 1).
    """
    idx = np.arange(n * n).reshape(n, n)
    perms = np.zeros((8, n * n), dtype=np.int64)
    for k in range(4):
        for reflect in range(2):
            cur = idx
            if k > 0:
                cur = np.rot90(cur, k=k, axes=(0, 1))
            if reflect:
                cur = np.flip(cur, axis=1)
            perms[k * 2 + reflect] = cur.reshape(-1)
    out = torch.from_numpy(perms).contiguous()
    if device is not None:
        out = out.to(device)
    return out


@functools.cache
def cell_perms_for(n_cells: int, device: torch.device) -> torch.Tensor:
    """Cached D4 cell permutations for a square grid of `n_cells` cells."""
    n_grid = int(round(n_cells ** 0.5))
    assert n_grid * n_grid == n_cells, f"dihedral augmentation needs a square grid, S={n_cells}"
    return build_dihedral_cell_perms(n=n_grid, device=device)


def sample_chain_augs(
    n_rows: int,
    n_channels: int,
    device: torch.device,
    with_dihedral: bool = True,
    vocab_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fresh per-row (digit_perm [N, C], dih_idx [N] or None).

    Only the first `vocab_dim` channels are permuted (identity on the rest;
    0 means no channel permutation).
    """
    vd = n_channels if vocab_dim is None else vocab_dim
    # argsort of uniform noise is a uniform random permutation per row.
    rand = torch.rand(n_rows, vd, device=device)
    vocab_perm = rand.argsort(dim=-1)
    if vd < n_channels:
        tail = torch.arange(vd, n_channels, device=device).unsqueeze(0).expand(n_rows, -1)
        digit_perm = torch.cat([vocab_perm, tail], dim=-1)
    else:
        digit_perm = vocab_perm
    dih_idx = (
        torch.randint(0, 8, (n_rows,), device=device)
        if with_dihedral else None
    )
    return digit_perm, dih_idx


def apply_aug_state(
    state: torch.Tensor,        # [N, S, C], any float dtype
    digit_perm: torch.Tensor,   # [N, C] long
    dih_idx: torch.Tensor | None,      # [N] long, or None for digit-perm-only
    cell_perms: torch.Tensor | None,   # [8, S] long, or None for digit-perm-only
) -> torch.Tensor:
    """Apply the digit permutation, then the dihedral one (skipped when cell_perms is None)."""
    N, S, C = state.shape
    out = state.gather(-1, digit_perm.unsqueeze(1).expand(N, S, C))
    if cell_perms is None or dih_idx is None:
        return out
    cell_perm = cell_perms[dih_idx]                              # [N, S]
    out = out.gather(1, cell_perm.unsqueeze(-1).expand(N, S, C))
    return out


def apply_aug_mask(
    mask: torch.Tensor,         # [N, S] (any dtype, typically bool)
    dih_idx: torch.Tensor | None,      # [N] long, or None for digit-perm-only
    cell_perms: torch.Tensor | None,   # [8, S] long, or None for digit-perm-only
) -> torch.Tensor:
    """Apply the dihedral permutation to a per-cell mask (identity when cell_perms is None)."""
    if cell_perms is None or dih_idx is None:
        return mask
    cell_perm = cell_perms[dih_idx]                              # [N, S]
    return mask.gather(1, cell_perm)


def invert_aug_state(
    state_aug: torch.Tensor,    # [N, S, C]
    digit_perm: torch.Tensor,   # [N, C] long
    dih_idx: torch.Tensor | None,      # [N] long, or None for digit-perm-only
    cell_perms: torch.Tensor | None,   # [8, S] long, or None for digit-perm-only
) -> torch.Tensor:
    """Undo `apply_aug_state` exactly."""
    N, S, C = state_aug.shape
    inv_digit = digit_perm.argsort(dim=-1)                       # [N, C]
    if cell_perms is None or dih_idx is None:
        return state_aug.gather(-1, inv_digit.unsqueeze(1).expand(N, S, C))
    cell_perm = cell_perms[dih_idx]                              # [N, S]
    inv_cell = cell_perm.argsort(dim=-1)                         # [N, S]
    out = state_aug.gather(1, inv_cell.unsqueeze(-1).expand(N, S, C))
    out = out.gather(-1, inv_digit.unsqueeze(1).expand(N, S, C))
    return out


@dataclass
class Frame:
    """A per-row relabelling of the batch; all None is the identity."""

    digit_perm: torch.Tensor | None = None     # [B, C]
    dih_idx: torch.Tensor | None = None        # [B]
    cell_perms: torch.Tensor | None = None     # [8, S]

    @classmethod
    def sample(cls, step: StepConfig, B: int, S: int, C: int, device: torch.device) -> Frame:
        """A fresh frame per `step.augment*`, or the identity when augmentation is off."""
        if not step.augment:
            return cls()
        cell_perms = cell_perms_for(S, device) if step.augment_dihedral else None
        vocab_dim = step.vocab_dim if step.permute_digits else 0
        digit_perm, dih_idx = sample_chain_augs(
            B, C, device, with_dihedral=step.augment_dihedral, vocab_dim=vocab_dim)
        return cls(digit_perm, dih_idx, cell_perms)

    def state(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None or self.digit_perm is None:
            return x
        return apply_aug_state(x, self.digit_perm, self.dih_idx, self.cell_perms)

    def mask(self, m: torch.Tensor | None) -> torch.Tensor | None:
        if m is None or self.digit_perm is None:
            return m
        return apply_aug_mask(m, self.dih_idx, self.cell_perms)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        if self.digit_perm is None:
            return x
        return invert_aug_state(x, self.digit_perm, self.dih_idx, self.cell_perms)


def aug_forward(model, state: torch.Tensor, given_mask: torch.Tensor, step: StepConfig, *,
                orig_y: torch.Tensor | None = None, carry_kw: dict | None = None):
    """Grad-tracked per-loop forward under a fresh frame, as `dpll_step` uses at no-grad time.

    Returns the model outputs (in the augmented frame) and the inputs mapped
    into that frame: {"aug_state", "aug_given_mask", "aug_orig_y"}.
    """
    frame = Frame.sample(step, *state.shape, state.device)
    state_aug = frame.state(state)
    out = model(state_aug, return_all=True, **(carry_kw or {}))
    return out, {
        "aug_state": state_aug,
        "aug_given_mask": frame.mask(given_mask),
        "aug_orig_y": frame.state(orig_y),
    }
