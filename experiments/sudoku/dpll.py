"""One DPLL-style step shared by the trainers and the solver.

Threshold deduction (one or more passes), conflict / solved detection, then
one branching decision per undecided row. With `StepConfig.augment`, each pass
runs under a fresh digit / dihedral relabelling that is undone before
returning, so callers only see the canonical frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from experiments.sudoku.aug import Frame

CellPolicy = Literal["uniform", "mrv", "min_entropy", "max_entropy"]
DigitPolicy = Literal["sample", "argmax", "rank_k"]


@dataclass
class StepConfig:
    threshold: float = 0.10       # eliminate a candidate when sigmoid(bce) < threshold
    cls_threshold: float = 0.5    # conflict head fires when sigmoid(logit) > cls_threshold
    temp_decide: float = 1.5      # softmax temperature for digit_policy="sample"
    augment: bool = False         # fresh digit / dihedral relabelling per pass
    augment_dihedral: bool = True     # include D4 (needs a square grid)
    vocab_dim: int | None = None      # leading editable channels; the rest are locked (None = all)
    permute_digits: bool = True       # include the channel permutation (off when labels differ)

    # Forward + eliminate passes before a round's single decision. "fixpoint"
    # repeats until a pass eliminates nothing, at most `deduce_pass_cap` times.
    deduce_passes: int | Literal["fixpoint"] = 1
    deduce_pass_cap: int = 16
    n_loops: int | None = None        # model loop count at inference (None = trained)

    # Which multi-candidate cell to branch on: uniformly at random, fewest
    # candidates (mrv), or lowest / highest entropy of the softmax head over
    # alive candidates. Ties break uniformly at random.
    cell_policy: CellPolicy = "uniform"
    # Which candidate to commit to: softmax sample at `temp_decide`, argmax, or
    # the k-th best for the k-th chain of a puzzle (rank_k).
    digit_policy: DigitPolicy = "sample"
    # Logits the step reads: the last internal iteration's, or their mean over iterations.
    readout: Literal["final", "mean"] = "final"

    def __post_init__(self):
        if self.cell_policy not in CELL_POLICIES:
            raise ValueError(f"cell_policy must be one of {sorted(CELL_POLICIES)}, "
                             f"got {self.cell_policy!r}")
        if self.digit_policy not in DIGIT_POLICIES:
            raise ValueError(f"digit_policy must be one of {sorted(DIGIT_POLICIES)}, "
                             f"got {self.digit_policy!r}")
        if not (self.deduce_passes == "fixpoint"
                or (isinstance(self.deduce_passes, int) and self.deduce_passes >= 1)):
            raise ValueError(f"deduce_passes must be >= 1 or 'fixpoint', "
                             f"got {self.deduce_passes!r}")

    @property
    def max_passes(self) -> int:
        return self.deduce_pass_cap if self.deduce_passes == "fixpoint" else self.deduce_passes


# ----- Cell policies: (multi [N, S] float, cd_state, cd_sm, vd) -> cell [N] -----

def _cell_uniform(multi, cd_state, cd_sm, vd):
    return torch.multinomial(multi, 1).squeeze(-1)


def _pick_extreme(score, multi, lowest: bool):
    """Uniformly random cell among the multi-alive cells with the lowest (or highest) score."""
    multi = multi > 0.5
    fill = float("inf") if lowest else float("-inf")
    score = torch.where(multi, score, torch.full_like(score, fill))
    if lowest:
        tie_set = multi & (score <= score.min(dim=-1, keepdim=True).values + 1e-6)
    else:
        tie_set = multi & (score >= score.max(dim=-1, keepdim=True).values - 1e-6)
    return torch.multinomial(tie_set.float(), 1).squeeze(-1)


def _alive_entropy(cd_state, cd_sm, vd):
    """[N, S] entropy of the softmax head renormalized over each cell's alive candidates."""
    C = cd_state.shape[-1]
    alive = cd_state > 0.5
    if vd < C:
        alive[..., vd:] = False
    logp = torch.log_softmax(cd_sm.masked_fill(~alive, float("-inf")), dim=-1)
    return -(logp.exp() * logp).nan_to_num(0.0).sum(dim=-1)


CELL_POLICIES = {
    "uniform": _cell_uniform,
    "mrv": lambda multi, cd_state, cd_sm, vd: _pick_extreme(
        cd_state[..., :vd].sum(dim=-1), multi, lowest=True),
    "min_entropy": lambda multi, cd_state, cd_sm, vd: _pick_extreme(
        _alive_entropy(cd_state, cd_sm, vd), multi, lowest=True),
    "max_entropy": lambda multi, cd_state, cd_sm, vd: _pick_extreme(
        _alive_entropy(cd_state, cd_sm, vd), multi, lowest=False),
}


# ----- Digit policies: (logits [N, C] with dead = -inf, cfg, rank [N]) -> digit [N] -----

def _digit_sample(logits, cfg, rank):
    if cfg.temp_decide > 0:
        return torch.multinomial(torch.softmax(logits / cfg.temp_decide, dim=-1), 1).squeeze(-1)
    return logits.argmax(dim=-1)


def _digit_rank_k(logits, cfg, rank):
    """The rank-th best alive candidate (clamped to the number alive)."""
    alive = torch.isfinite(logits).sum(dim=-1).clamp(min=1)
    rank = torch.minimum(rank.clamp(min=0), alive - 1)
    order = logits.argsort(dim=-1, descending=True)
    return order.gather(-1, rank.unsqueeze(-1)).squeeze(-1)


DIGIT_POLICIES = {
    "sample": _digit_sample,
    "argmax": lambda logits, cfg, rank: logits.argmax(dim=-1),
    "rank_k": _digit_rank_k,
}


def dpll_step(
    model,
    state: torch.Tensor,
    given_mask: torch.Tensor,
    cfg: StepConfig,
    *,
    in_puzzle_mask: torch.Tensor | None = None,
    want_stats: bool = True,
    decide_rank: torch.Tensor | None = None,
    carry_kw: dict | None = None,
):
    """Deduce, check for conflict / solution, then branch once per undecided row.

    Args:
        state: [B, S, C] lattice state, canonical frame.
        given_mask: [B, S] cells fixed by the puzzle.
        in_puzzle_mask: [B, S] cells that belong to the puzzle (None = all).
        want_stats: Add the `n_*` counts to `info` (each forces a GPU sync).
        decide_rank: [B] candidate rank per row, for digit_policy="rank_k".
        carry_kw: `orig_x` / `carry` for latent-carry models; needs
            augment=False and a single pass.

    Returns:
        (new_state, conflict, solved, out, info). `new_state` is canonical;
        `out` holds the last pass's model outputs. `info` has `deduce_mask` (the
        union over passes, canonical), `n_passes`, `pass_deduce_masks` (one per
        pass, multi-pass only), `carry_out` for carry models, and the stats.
    """
    B, S, C = state.shape
    device = state.device
    vd = cfg.vocab_dim if cfg.vocab_dim is not None else C
    model_kw = {} if cfg.n_loops is None else {"n_loops": cfg.n_loops}
    if carry_kw:
        if cfg.augment or cfg.max_passes != 1:
            raise ValueError("latent carry needs augment=False and a single deduction pass")
        model_kw.update(carry_kw)

    max_passes = cfg.max_passes
    cur_state = state
    deduce_union = None
    prev_status = None
    pass_masks: list[torch.Tensor] = []
    # Multi-pass: a row is frozen once a pass finds it solved or in conflict.
    done = torch.zeros(B, dtype=torch.bool, device=device)
    was_done = done
    for n_passes in range(1, max_passes + 1):
        frame = Frame.sample(cfg, B, S, C, device)
        state_aug = frame.state(cur_state)
        gm_aug = frame.mask(given_mask)
        ip_aug = frame.mask(in_puzzle_mask)

        out = model(state_aug, use_final=cfg.readout == "final", **model_kw)
        sm_logits = out["softmax"]

        deduce_mask_aug = (torch.sigmoid(out["bce"]) < cfg.threshold) & (state_aug > 0.5)
        deduce_mask_aug = deduce_mask_aug & ~gm_aug.unsqueeze(-1)
        if ip_aug is not None:
            deduce_mask_aug = deduce_mask_aug & ip_aug.unsqueeze(-1)
        deduce_mask_aug[..., vd:] = False       # locked channels are never eliminated
        new_state_aug = state_aug.masked_fill(deduce_mask_aug, 0.0)

        # Status after deduction; only editable channels and in-puzzle cells count.
        n_alive = new_state_aug[..., :vd].sum(dim=-1)            # [B, S]
        if ip_aug is not None:
            empty_cell = ((n_alive == 0) & ip_aug).any(dim=-1)
            all_singleton = ((n_alive == 1) | ~ip_aug).all(dim=-1)
        else:
            empty_cell = (n_alive == 0).any(dim=-1)
            all_singleton = (n_alive == 1).all(dim=-1)
        cls_fires = torch.zeros_like(empty_cell)
        if "conflict" in out:
            cls_fires = torch.sigmoid(out["conflict"]).squeeze(-1) > cfg.cls_threshold
        conflict = empty_cell | cls_fires
        solved = all_singleton & ~conflict

        if max_passes == 1:
            break
        was_done = done
        if n_passes > 1:
            conflict, solved, empty_cell, cls_fires = (
                torch.where(was_done, old, new) for old, new in
                zip(prev_status, (conflict, solved, empty_cell, cls_fires)))
        prev_status = (conflict, solved, empty_cell, cls_fires)
        pass_mask = frame.invert(deduce_mask_aug.float()) > 0.5 if cfg.augment else deduce_mask_aug
        pass_mask = pass_mask & ~was_done[:, None, None]
        cur_state = torch.where(was_done[:, None, None], cur_state, frame.invert(new_state_aug))
        pass_masks.append(pass_mask)
        deduce_union = pass_mask if deduce_union is None else deduce_union | pass_mask
        done = conflict | solved
        if n_passes == max_passes:
            break
        if bool(done.all()) or not bool((pass_mask & ~done[:, None, None]).any()):
            break
    can_decide = ~conflict & ~solved

    if can_decide.any():
        cd_b_idx = can_decide.nonzero(as_tuple=True)[0]              # [N_cd]
        cd_state = new_state_aug[cd_b_idx]                           # [N_cd, S, C]
        cd_multi_alive = (cd_state[..., :vd].sum(dim=-1) > 1.5).float()  # [N_cd, S]
        if ip_aug is not None:
            cd_multi_alive = cd_multi_alive * ip_aug[cd_b_idx].float()   # never branch outside the puzzle
        cell_idx = CELL_POLICIES[cfg.cell_policy](
            cd_multi_alive, cd_state, sm_logits[cd_b_idx], vd)

        N_cd = cd_b_idx.shape[0]
        cd_arange = torch.arange(N_cd, device=device)
        sm_at_cell = sm_logits[cd_b_idx, cell_idx]                   # [N_cd, C]
        alive_at_cell = cd_state[cd_arange, cell_idx] > 0.5          # [N_cd, C]
        sm_at_cell = sm_at_cell.masked_fill(~alive_at_cell, float("-inf"))
        sm_at_cell[..., vd:] = float("-inf")
        rank = (decide_rank[cd_b_idx].long() if decide_rank is not None
                else torch.zeros(N_cd, dtype=torch.long, device=device))
        digit = DIGIT_POLICIES[cfg.digit_policy](sm_at_cell, cfg, rank)

        new_state_aug[cd_b_idx, cell_idx, :vd] = 0.0
        new_state_aug[cd_b_idx, cell_idx, digit] = 1.0

    new_state = frame.invert(new_state_aug)
    if max_passes > 1:
        new_state = torch.where(was_done[:, None, None], cur_state, new_state)
    if deduce_union is not None:
        deduce_mask = deduce_union
    else:
        deduce_mask = frame.invert(deduce_mask_aug.float()) > 0.5 if cfg.augment else deduce_mask_aug

    info = {"deduce_mask": deduce_mask, "n_passes": n_passes, "pass_deduce_masks": pass_masks}
    if "carry_out" in out:
        info["carry_out"] = out["carry_out"].detach()
    if want_stats:
        info.update({
            "n_deduced": int(deduce_mask.sum().item()),
            "n_decided": int(can_decide.sum().item()),
            "n_conflict": int(conflict.sum().item()),
            "n_conflict_empty": int(empty_cell.sum().item()),
            "n_conflict_cls": int(cls_fires.sum().item()),
            "n_solved": int(solved.sum().item()),
        })
    return new_state, conflict, solved, out, info
