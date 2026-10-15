"""Unified DPLL-style step for sudoku: unit propagation
(threshold deduction) + branching (decision) + status / conflict
detection, all in a single function.

Same function called by both the trainer and the solver.

Deduction is always deterministic threshold (kill if σ(bce) < threshold).
The eval-time `solve()` never passes `orig_y`; the operator is
deterministic at eval automatically.

Decision: pick a multi-alive cell uniformly, softmax-sample one digit
at `temp_decide`, pin that cell to that digit. Monotonic on the chosen
cell — only a chain reset undoes a commit. Runs every round on
`~conflict & ~solved` batch elements; conflict / solved elements pass
through unchanged so the caller can reset / accept them.

Augmentation (`cfg.augment`):
  Per-row dataset-style aug (digit perm + D4 dihedral) applied here
  inside `dpll_step` as a black-box wrapper around the forward +
  deduce + decide pipeline:
    1. Sample fresh per-row aug.
    2. Apply to `state`, `given_mask`, and `orig_y` (if present).
    3. Forward + status + decision all in the augmented frame.
    4. Invert the aug on `new_state` and on `info["deduce_mask"]` before
       returning, so the caller (solve / train) sees canonical-frame
       outputs throughout.
  The aug is bijective on cells and digits, so all booleans (conflict,
  solved, empty-cell, gt_conflict) are frame-invariant — only positional
  things need explicit invert.

  The trainer also needs to do its own grad-tracked forward; for that,
  use `aug_forward()` in `aug.py` with `cfg.augment` so the grad-side
  forward also operates on aug-frame state and trains invariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.sudoku.aug import (
    apply_aug_mask,
    apply_aug_state,
    build_dihedral_cell_perms,
    invert_aug_state,
    sample_chain_augs,
)


@dataclass
class StepConfig:
    threshold: float = 0.10       # deterministic deduction threshold (always on)
    cls_threshold: float = 0.5    # sigmoid cutoff for conflict head firing
    temp_decide: float = 1.5      # softmax temperature for the per-round digit pick
    # Per-row dataset-style aug applied fresh inside every `dpll_step`
    # call — and (when used by `aug_forward`) inside the trainer's
    # grad-tracked forward as well. Default False; the trainer/solver
    # entrypoints explicitly opt in via their `--augment` flag.
    augment: bool = False
    # Whether the spatial dihedral (D4) component of `augment` is included.
    # Default True = sudoku behavior (digit perm + dihedral). Set False for
    # snowflake: the covering grid is hex (S not a perfect square) and
    # dihedral on the covering would not preserve hex topology — only digit
    # perm remains a valid symmetry. Inert when augment=False.
    augment_dihedral: bool = True
    # Number of "vocab" channels at the start of the channel dim (state[..., :vocab_dim]).
    # Channels at index >= vocab_dim are treated as locked auxiliary inputs and
    # never modified by deduce/decide. None = treat all C channels as vocab
    # (sudoku behavior). Snowflake sets vocab_dim=6: a 7th channel carries the
    # in-puzzle mask which the operator must not touch.
    vocab_dim: int | None = None
    # Whether the digit-perm component of `augment` shuffles channels. Default
    # True = sudoku/snowflake behavior (digit-perm permutes the first vocab_dim
    # channels). Set False when channel labels aren't interchangeable — e.g.
    # maze where the 5 channels (wall/free/S/G/path) carry distinct
    # semantics and shuffling them would break the data. Inert when
    # `augment=False`.
    permute_digits: bool = True

    # ----- Deduction -----
    # Forward + threshold-eliminate passes on the same state before the one
    # decision of a round. "fixpoint" repeats until a pass eliminates nothing
    # (or every row is in conflict / solved), at most `deduce_pass_cap` times.
    deduce_passes: int | str = 1
    deduce_pass_cap: int = 16
    # Model loop count at inference (None = the trained `n_loops`).
    n_loops: int | None = None

    # ----- Decision -----
    # Which multi-candidate cell to branch on:
    #   uniform     — uniformly at random
    #   mrv         — fewest remaining candidates
    #   min_entropy / max_entropy — of the softmax head over alive candidates
    # Deterministic choices break ties uniformly at random.
    cell_policy: str = "uniform"
    # Which candidate to commit to at that cell:
    #   sample  — softmax head at `temp_decide` (argmax if temp_decide <= 0)
    #   argmax  — highest softmax logit
    #   rank_k  — the k-th best candidate for the k-th chain of a puzzle
    #             (`decide_rank`; rank 0 = argmax when no rank is given)
    digit_policy: str = "sample"

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


# Cache of per-(n, device) cell permutations so we don't rebuild them on
# every dpll_step call. Keyed by (n, device.type, device.index).
_CELL_PERMS_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def _get_cell_perms(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device.type, device.index)
    cached = _CELL_PERMS_CACHE.get(key)
    if cached is None:
        cached = build_dihedral_cell_perms(n=n, device=device)
        _CELL_PERMS_CACHE[key] = cached
    return cached


def _sample_frame(cfg: StepConfig, state, given_mask, orig_y, in_puzzle_mask):
    """Fresh per-row augmentation (or the identity) and the inputs mapped
    into that frame: (digit_perm, dih_idx, cell_perms, state, given_mask,
    orig_y, in_puzzle_mask)."""
    if not cfg.augment:
        return None, None, None, state, given_mask, orig_y, in_puzzle_mask
    B, S, C = state.shape
    device = state.device
    if cfg.augment_dihedral:
        n_grid = int(round(S ** 0.5))
        assert n_grid * n_grid == S, (
            f"augment_dihedral=True but S={S} is not a perfect square"
        )
        cell_perms = _get_cell_perms(n_grid, device)
    else:
        cell_perms = None
    # `permute_digits=False` overrides cfg.vocab_dim with 0, yielding an
    # identity channel-perm (digit-perm becomes a no-op). Used by maze.
    aug_vd = 0 if not cfg.permute_digits else cfg.vocab_dim
    digit_perm, dih_idx = sample_chain_augs(
        B, C, device, with_dihedral=cfg.augment_dihedral, vocab_dim=aug_vd,
    )
    return (
        digit_perm, dih_idx, cell_perms,
        apply_aug_state(state, digit_perm, dih_idx, cell_perms),
        apply_aug_mask(given_mask, dih_idx, cell_perms),
        apply_aug_state(orig_y, digit_perm, dih_idx, cell_perms) if orig_y is not None else None,
        apply_aug_mask(in_puzzle_mask, dih_idx, cell_perms) if in_puzzle_mask is not None else None,
    )


# ----- Cell policies: (multi [N, S] float, cd_state, cd_sm, vd) -> cell [N] -----

def _cell_uniform(multi, cd_state, cd_sm, vd):
    return torch.multinomial(multi, 1).squeeze(-1)


def _pick_extreme(score, multi, lowest: bool):
    """Uniformly random cell among the multi-alive cells with the lowest
    (or highest) score."""
    multi = multi > 0.5
    fill = float("inf") if lowest else float("-inf")
    score = torch.where(multi, score, torch.full_like(score, fill))
    if lowest:
        tie_set = multi & (score <= score.min(dim=-1, keepdim=True).values + 1e-6)
    else:
        tie_set = multi & (score >= score.max(dim=-1, keepdim=True).values - 1e-6)
    return torch.multinomial(tie_set.float(), 1).squeeze(-1)


def _alive_entropy(cd_state, cd_sm, vd):
    """Entropy of the softmax head renormalized over each cell's alive
    vocab candidates. [N, S]"""
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
    state: torch.Tensor,        # [B, S, C], float, CANONICAL
    given_mask: torch.Tensor,   # [B, S], bool, CANONICAL (cells pinned by puzzle givens)
    cfg: StepConfig,
    *,
    orig_y: torch.Tensor | None = None,   # [B, S, C] one-hot GT; train-only; CANONICAL
    in_puzzle_mask: torch.Tensor | None = None,   # [B, S] bool, CANONICAL — cells active in this puzzle
    want_stats: bool = True,
    decide_rank: torch.Tensor | None = None,   # [B] long, for digit_policy="rank_k"
    carry_kw: dict | None = None,              # orig_x / carry for latent-carry models
):
    """One unified DPLL-style step (unit propagation + branching +
    conflict detection). Returns (new_state, conflict, solved, out, info).

    Inputs are CANONICAL (original-puzzle frame); outputs are likewise
    canonical (`new_state`, `info["deduce_mask"]` inverted before
    return). `out` (model logits) is in the AUGMENTED frame of the last
    deduction pass — callers that need it for losses must use
    `info["aug_state"]`, `info["aug_given_mask"]`, `info["aug_orig_y"]` to
    match the frame.

    Always deterministic threshold deduction. `orig_y` is unused by the
    operator itself, but the parameter is retained so callers / aug
    helpers can permute it under the chosen frame for the trainer's grad
    path. Pass `None` if you don't need to track GT under augmentation.

    With `cfg.deduce_passes` > 1 (or "fixpoint"), forward + eliminate is
    repeated on the same state, each pass under a fresh augmentation, before
    the single decision; the loop ends early once a pass eliminates nothing
    or every row is in conflict or solved. Conflict / solved / decision come
    from the last pass, `info["deduce_mask"]` is the union of all passes'
    eliminations, `info["n_passes"]` counts the forwards spent, and
    `info["pass_deduce_masks"]` lists each pass's eliminations (multi-pass
    only).

    `carry_kw` (latent-carry models: the original puzzle `orig_x` and the
    previous step's `carry`) is passed to the model, and the model's
    `carry_out` is returned as `info["carry_out"]`. A carried latent lives in
    a fixed frame, so this needs `augment=False` and a single pass.

    `want_stats` (default True): when True, `info` contains the full diagnostic
    dict — `n_deduced`, `n_decided`, `n_conflict`, `n_conflict_empty`,
    `n_conflict_cls`, `n_solved`, `deduce_mask`. Each scalar count requires a
    `.item()` call which forces a CPU-GPU sync. When False, only `deduce_mask`
    (a tensor reference, free) is populated; the `.item()` calls are skipped
    entirely. Pass `want_stats=False` from hot paths that don't print
    diagnostics on every call (e.g. `solve()` per chain-round, training steps
    that aren't on `log_every`).
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
    pass_masks: list[torch.Tensor] = []
    for n_passes in range(1, max_passes + 1):
        (digit_perm, dih_idx, cell_perms,
         state_aug, gm_aug, orig_y_aug, ip_aug) = _sample_frame(
            cfg, cur_state, given_mask, orig_y, in_puzzle_mask)

        out = model(state_aug, use_final=True, **model_kw)
        bce_logits = out["bce"]
        sm_logits = out["softmax"]

        # ===== Deduction (deterministic threshold) — in aug frame =====
        probs = torch.sigmoid(bce_logits)
        deduce_mask_aug = (probs < cfg.threshold) & (state_aug > 0.5)
        deduce_mask_aug = deduce_mask_aug & ~gm_aug.unsqueeze(-1)
        if ip_aug is not None:
            # Don't deduce on out-of-puzzle cells (they're permanently zero anyway,
            # but explicit gating keeps soundness diagnostics clean).
            deduce_mask_aug = deduce_mask_aug & ip_aug.unsqueeze(-1)
        if vd < C:
            # Don't touch the auxiliary (post-vocab) channels — e.g. snowflake's
            # locked mask channel. Build an explicit vocab-only mask and AND it in.
            vocab_mask = torch.zeros(C, dtype=torch.bool, device=device)
            vocab_mask[:vd] = True
            deduce_mask_aug = deduce_mask_aug & vocab_mask

        new_state_aug = state_aug.masked_fill(deduce_mask_aug, 0.0)

        # ===== Status (post-deduce, pre-decide) — in aug frame =====
        # Count alive bits over vocab channels only (auxiliary channels are
        # always-on locks — counting them would double-count and break the
        # singleton check).
        n_alive = new_state_aug[..., :vd].sum(dim=-1)            # [B, S]
        if ip_aug is not None:
            # Only in-puzzle cells participate in empty / singleton checks.
            # Out-of-puzzle cells have all-zero state by construction; ignoring
            # them avoids false empty_cell / false-not-all-singleton firings.
            empty_cell = ((n_alive == 0) & ip_aug).any(dim=-1)
            all_singleton = ((n_alive == 1) | ~ip_aug).all(dim=-1)
        else:
            empty_cell = (n_alive == 0).any(dim=-1)        # [B] — soundness-head collapse
            all_singleton = (n_alive == 1).all(dim=-1)     # [B] (frame-invariant boolean)
        cls_fires = torch.zeros_like(empty_cell)
        if "conflict" in out:
            cls_sigmoid = torch.sigmoid(out["conflict"]).squeeze(-1)
            cls_fires = cls_sigmoid > cfg.cls_threshold
        conflict = empty_cell | cls_fires           # [B] (frame-invariant boolean)
        solved = all_singleton & ~conflict

        if max_passes == 1:
            break
        # Multi-pass: carry the deduced state (canonical) into the next pass.
        if cfg.augment:
            pass_mask = invert_aug_state(
                deduce_mask_aug.float(), digit_perm, dih_idx, cell_perms) > 0.5
            cur_state = invert_aug_state(new_state_aug, digit_perm, dih_idx, cell_perms)
        else:
            pass_mask = deduce_mask_aug
            cur_state = new_state_aug
        pass_masks.append(pass_mask)
        deduce_union = pass_mask if deduce_union is None else deduce_union | pass_mask
        if n_passes == max_passes:
            break
        if bool((conflict | all_singleton).all()) or not bool(pass_mask.any()):
            break
    can_decide = ~conflict & ~solved            # [B]

    # ===== Decision: pick a multi-alive cell, pin one candidate (aug frame) =====
    if can_decide.any():
        cd_b_idx = can_decide.nonzero(as_tuple=True)[0]              # [N_cd]
        cd_state = new_state_aug[cd_b_idx]                           # [N_cd, S, C]
        # Multi-alive over vocab channels only (so an in-puzzle cell with
        # one vocab bit alive + locked mask isn't counted as multi-alive).
        cd_multi_alive = (cd_state[..., :vd].sum(dim=-1) > 1.5).float()  # [N_cd, S]
        if ip_aug is not None:
            # Mask out out-of-puzzle cells from the decision pool — they
            # have sum>1.5 only by accident, but explicit gate is safer.
            cd_multi_alive = cd_multi_alive * ip_aug[cd_b_idx].float()
        cell_idx = CELL_POLICIES[cfg.cell_policy](
            cd_multi_alive, cd_state, sm_logits[cd_b_idx], vd)

        N_cd = cd_b_idx.shape[0]
        cd_arange = torch.arange(N_cd, device=device)
        sm_at_cell = sm_logits[cd_b_idx, cell_idx]                   # [N_cd, C]
        alive_at_cell = cd_state[cd_arange, cell_idx] > 0.5          # [N_cd, C]
        sm_at_cell = sm_at_cell.masked_fill(~alive_at_cell, float("-inf"))
        if vd < C:
            # Don't ever pin to a non-vocab channel.
            sm_at_cell[..., vd:] = float("-inf")
        rank = (decide_rank[cd_b_idx].long() if decide_rank is not None
                else torch.zeros(N_cd, dtype=torch.long, device=device))
        digit = DIGIT_POLICIES[cfg.digit_policy](sm_at_cell, cfg, rank)

        # Zero only the vocab channels — auxiliary channels (e.g. snowflake's
        # locked mask channel) must not be touched by the decision step.
        if vd < C:
            new_state_aug[cd_b_idx, cell_idx, :vd] = 0.0
        else:
            new_state_aug[cd_b_idx, cell_idx] = 0.0
        new_state_aug[cd_b_idx, cell_idx, digit] = 1.0

    # ===== Invert aug on new_state and deduce_mask before returning =====
    if cfg.augment:
        new_state = invert_aug_state(new_state_aug, digit_perm, dih_idx, cell_perms)
        deduce_mask = deduce_union if deduce_union is not None else invert_aug_state(
            deduce_mask_aug.float(), digit_perm, dih_idx, cell_perms,
        ) > 0.5
    else:
        new_state = new_state_aug
        deduce_mask = deduce_union if deduce_union is not None else deduce_mask_aug

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
    # Always expose aug params and aug-frame inputs (of the last pass)
    # alongside info — the trainer's grad forward typically uses
    # `aug_forward()` directly to get its own aug, but exposing the no-grad
    # aug here keeps the contract honest if a caller wants to inspect it.
    info["digit_perm"] = digit_perm
    info["dih_idx"] = dih_idx
    info["aug_state"] = state_aug
    info["aug_given_mask"] = gm_aug
    info["aug_orig_y"] = orig_y_aug
    info["aug_in_puzzle_mask"] = ip_aug
    return new_state, conflict, solved, out, info
