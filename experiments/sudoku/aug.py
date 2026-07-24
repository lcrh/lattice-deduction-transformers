"""Per-chain dataset-style augmentation for the solve loop.

Mirrors `lattice_diffusion/data/sudoku_extreme.py::_augment` but in
batched torch instead of per-sample numpy:

  - `augment_digit_perm`: permute the 9 digit channels (channel-dim).
  - `augment_dihedral`: full D4 (4 rotations × {identity, reflection})
    on the n×n cell grid (cell-dim, with n = sqrt(S)).

The purpose of moving this into the solver is search diversity: 64 chains
per puzzle currently start identical. With per-chain aug, each chain
explores a different "view" of the same puzzle — the symmetry group
guarantees these are equivalent problems, so any chain that solves its
augmented view yields a solution we can de-augment back to the original
frame.

Conventions used by the apply/invert pair:
  - We apply digit perm first, then spatial dihedral (matches dataset
    `_augment`). Order is irrelevant for correctness as long as the
    inverse applies them in reverse — which `invert_aug_state` does.
  - `digit_perm[b, c_new] = c_old` ⇒ apply does
    `out[..., c_new] = state[..., digit_perm[b, c_new]]`
    (i.e. gather along the channel dim with `digit_perm` as index).
  - `cell_perms[g, s_new] = s_old` ⇒ apply does
    `out[s_new, ...] = state[cell_perms[g, s_new], ...]`.
  - Inverse uses `argsort` of the per-row permutation, which is the
    integer inverse for a permutation tensor.
"""

from __future__ import annotations

import numpy as np
import torch


def build_dihedral_cell_perms(n: int = 9, device: torch.device | None = None) -> torch.Tensor:
    """Return [8, n*n] long: cell permutations for the 8 D4 elements.

    Group element index = `k_rot * 2 + reflect_flag` (k_rot in [0, 4),
    reflect_flag in {0, 1}). Reflection is the same one the dataset uses
    (np.flip on axis=1, i.e. left-right mirror in row-major (i, j)).
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


def sample_chain_augs(
    n_rows: int,
    n_digits: int,
    device: torch.device,
    with_dihedral: bool = True,
    vocab_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample fresh per-row augmentations.

    Returns:
      digit_perm: [n_rows, n_digits] long — random permutation per row,
        where digit_perm[r, c_new] = c_old. When `vocab_dim < n_digits`
        is provided, only the first `vocab_dim` channels are shuffled
        (random permutation over [0, vocab_dim)) and the trailing
        channels are mapped by identity. This is for snowflake's
        n_channels=7 (vocab=6 + 1 locked mask channel) where shuffling
        the mask channel under digit-perm would scramble the puzzle.
        `vocab_dim=0` is the degenerate case: identity over all channels
        (no channel perm at all) — used by maze where the channel
        labels (wall/free/start/goal/path) aren't interchangeable.
      dih_idx: [n_rows] long — random group element index in [0, 8), or
        `None` if `with_dihedral=False` (callers pass this `None` straight
        through to `apply_aug_state(..., cell_perms=None)` to skip the
        spatial step).
    """
    vd = n_digits if vocab_dim is None else vocab_dim
    # argsort of independent uniform random vectors gives a uniform
    # random permutation per row.
    rand = torch.rand(n_rows, vd, device=device)
    vocab_perm = rand.argsort(dim=-1)
    if vd < n_digits:
        # Identity on auxiliary (non-vocab) channels.
        tail = torch.arange(vd, n_digits, device=device).unsqueeze(0).expand(n_rows, -1)
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
    """Apply digit perm then spatial dihedral. Same order as dataset.

    `cell_perms is None` (or equivalently `dih_idx is None`) skips the
    spatial dihedral step — used by snowflake where the
    covering grid is hex (S not a perfect square) and the only valid
    aug is digit perm.
    """
    N, S, C = state.shape
    # Digit perm.
    out = state.gather(-1, digit_perm.unsqueeze(1).expand(N, S, C))
    if cell_perms is None or dih_idx is None:
        return out
    # Spatial dihedral.
    cell_perm = cell_perms[dih_idx]                              # [N, S]
    out = out.gather(1, cell_perm.unsqueeze(-1).expand(N, S, C))
    return out


def apply_aug_mask(
    mask: torch.Tensor,         # [N, S] (any dtype, typically bool)
    dih_idx: torch.Tensor | None,      # [N] long, or None for digit-perm-only
    cell_perms: torch.Tensor | None,   # [8, S] long, or None for digit-perm-only
) -> torch.Tensor:
    """Spatial-only — `given_mask` has no channel dim. With `cell_perms=None`
    the mask is unchanged (digit-perm doesn't touch positional masks)."""
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
    """Undo `apply_aug_state` exactly.

    apply did: digit-perm gather, then cell-perm gather.
    invert undoes them in reverse order, gathering with `argsort`-inverse
    of each per-row permutation. With `cell_perms=None` only the digit-
    perm inverse runs (matching `apply_aug_state` in the same mode).
    """
    N, S, C = state_aug.shape
    inv_digit = digit_perm.argsort(dim=-1)                       # [N, C]
    if cell_perms is None or dih_idx is None:
        # Only digit-perm was applied; invert it.
        return state_aug.gather(-1, inv_digit.unsqueeze(1).expand(N, S, C))
    cell_perm = cell_perms[dih_idx]                              # [N, S]
    inv_cell = cell_perm.argsort(dim=-1)                         # [N, S]

    # Undo cell-perm first (it was the last gather applied).
    out = state_aug.gather(1, inv_cell.unsqueeze(-1).expand(N, S, C))
    # Then undo digit-perm.
    out = out.gather(-1, inv_digit.unsqueeze(1).expand(N, S, C))
    return out


def aug_forward(
    model,
    state: torch.Tensor,           # [B, S, C] CANONICAL
    given_mask: torch.Tensor,      # [B, S]    CANONICAL
    *,
    orig_x: torch.Tensor | None = None,   # [B, S, C] immutable puzzle
    carry: torch.Tensor | None = None,    # [B, S, dim] fixed-frame latent
    orig_y: torch.Tensor | None = None,   # [B, S, C] CANONICAL or None
    augment: bool = True,
    augment_dihedral: bool = True,        # if False, digit-perm only (no spatial)
    cell_perms: torch.Tensor | None = None,  # optional preallocated [8, S]; ignored when augment_dihedral=False
    vocab_dim: int | None = None,         # see sample_chain_augs; permutes only first vocab_dim channels
    return_all: bool = False,
):
    """Sample fresh per-row aug, apply to inputs, run model forward.

    Used by the trainer's grad-tracked forward so the loss is computed
    in the augmented frame against augmented targets — exactly matching
    the operator that `dpll_step` applies at no-grad / eval time.

    `augment_dihedral=False` skips the spatial dihedral step entirely
    (snowflake covering grid isn't square; only digit-perm is a valid
    symmetry).

    Returns:
      out: model output dict (logits in AUGMENTED frame).
      aug_info: dict with
        "digit_perm": [B, C] long or None
        "dih_idx":    [B] long or None
        "aug_state":  [B, S, C] (the input the model saw, aug frame)
        "aug_given_mask": [B, S]
        "aug_orig_y": [B, S, C] or None
    """
    B, S, C = state.shape
    device = state.device
    if augment:
        if augment_dihedral:
            if cell_perms is None:
                n_grid = int(round(S ** 0.5))
                assert n_grid * n_grid == S, (
                    f"augment_dihedral=True but S={S} is not a perfect square"
                )
                cell_perms = build_dihedral_cell_perms(n=n_grid, device=device)
        else:
            cell_perms = None
        digit_perm, dih_idx = sample_chain_augs(
            B, C, device, with_dihedral=augment_dihedral, vocab_dim=vocab_dim,
        )
        state_aug = apply_aug_state(state, digit_perm, dih_idx, cell_perms)
        gm_aug = apply_aug_mask(given_mask, dih_idx, cell_perms)
        orig_y_aug = (
            apply_aug_state(orig_y, digit_perm, dih_idx, cell_perms)
            if orig_y is not None else None
        )
        orig_x_aug = (
            apply_aug_state(orig_x, digit_perm, dih_idx, cell_perms)
            if orig_x is not None else None
        )
    else:
        digit_perm = None
        dih_idx = None
        state_aug = state
        gm_aug = given_mask
        orig_y_aug = orig_y
        orig_x_aug = orig_x

    if orig_x is None and carry is None:
        out = model(state_aug, return_all=return_all)
    else:
        if augment and carry is not None:
            raise ValueError(
                "cannot re-augment an opaque carried latent; use a fixed frame"
            )
        out = model(
            state_aug,
            return_all=return_all,
            orig_x=orig_x_aug,
            carry=carry,
        )
    aug_info = {
        "digit_perm": digit_perm,
        "dih_idx": dih_idx,
        "aug_state": state_aug,
        "aug_given_mask": gm_aug,
        "aug_orig_y": orig_y_aug,
        "aug_orig_x": orig_x_aug,
    }
    return out, aug_info
