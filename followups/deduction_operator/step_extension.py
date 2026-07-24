"""Extended dpll_step used by followup Modal eval/train entrypoints.

Core `experiments.sudoku.dpll.dpll_step` stays on the legacy single-pass path.
Followups attach an `ExtendedStep` instance via `StepConfig._extension`
(metadata public=False) so multi-pass deduction + decision policies live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.sudoku.aug import (
    apply_aug_mask,
    apply_aug_state,
    invert_aug_state,
    sample_chain_augs,
)
from experiments.sudoku.dpll import StepConfig, _get_cell_perms


@dataclass
class StepExtensionOpts:
    """Followup-only operating-point overrides for the extended step."""
    deduce_passes: int = 1
    deduce_pass_cap: int = 16
    cell_policy: str = "uniform"
    digit_policy: str = "softmax"


@dataclass
class ExtendedStep:
    """Callable drop-in for `StepConfig._extension`."""
    opts: StepExtensionOpts = field(default_factory=StepExtensionOpts)

    def __call__(self, model, state, given_mask, cfg: StepConfig, *,
                 orig_y=None, in_puzzle_mask=None, want_stats=True,
                 decide_rank=None):
        return extended_dpll_step(
            model, state, given_mask, cfg, self.opts,
            orig_y=orig_y, in_puzzle_mask=in_puzzle_mask,
            want_stats=want_stats, decide_rank=decide_rank,
        )


def make_extended_step(
    *,
    deduce_passes: int = 1,
    deduce_pass_cap: int = 16,
    cell_policy: str = "uniform",
    digit_policy: str = "softmax",
) -> ExtendedStep | None:
    """Return None when all defaults (core legacy path is enough)."""
    if (deduce_passes == 1 and cell_policy == "uniform"
            and digit_policy == "softmax"):
        return None
    return ExtendedStep(StepExtensionOpts(
        deduce_passes=deduce_passes,
        deduce_pass_cap=deduce_pass_cap,
        cell_policy=cell_policy,
        digit_policy=digit_policy,
    ))


def attach_step_extension(cfg: StepConfig, ext: ExtendedStep | None) -> StepConfig:
    cfg._extension = ext
    return cfg

def extended_dpll_step(
    model,
    state: torch.Tensor,        # [B, S, C], float, CANONICAL
    given_mask: torch.Tensor,   # [B, S], bool, CANONICAL (cells pinned by puzzle givens)
    cfg: StepConfig,
    opts: StepExtensionOpts,
    *,
    orig_y: torch.Tensor | None = None,   # [B, S, C] one-hot GT; train-only; CANONICAL
    in_puzzle_mask: torch.Tensor | None = None,   # [B, S] bool, CANONICAL — cells active in this puzzle
    want_stats: bool = True,
    decide_rank: torch.Tensor | None = None,   # [B] long — per-row rank for digit_policy="rank_k"
):
    """One unified DPLL-style step (unit propagation + branching +
    conflict detection). Returns (new_state, conflict, solved, out, info).

    Inputs are CANONICAL (original-puzzle frame); outputs are likewise
    canonical (`new_state`, `info["deduce_mask"]` inverted before
    return). `out` (model logits) is in the AUGMENTED frame — callers
    that need it for losses must use `info["aug_state"]`,
    `info["aug_given_mask"]`, `info["aug_orig_y"]` to match the frame.

    Always deterministic threshold deduction. `orig_y` is unused by the
    operator itself, but the parameter is retained so callers / aug
    helpers can permute it under the chosen frame for the trainer's grad
    path. Pass `None` if you don't need to track GT under augmentation.

    `want_stats` (default True): when True, `info` contains the full diagnostic
    dict — `n_deduced`, `n_decided`, `n_conflict`, `n_conflict_empty`,
    `n_conflict_cls`, `n_solved`, `deduce_mask`. Each scalar count requires a
    `.item()` call which forces a CPU-GPU sync. When False, only `deduce_mask`
    (a tensor reference, free) is populated; the `.item()` calls are skipped
    entirely. Pass `want_stats=False` from hot paths that don't print
    diagnostics on every call (e.g. `solve()` per chain-round, training steps
    that aren't on `log_every`).

    Multi-pass deduction (`opts.deduce_passes != 1`, ): the forward +
    threshold-eliminate is applied `deduce_passes` times (or to fixpoint when
    0, capped at `opts.deduce_pass_cap`) on the SAME state before the single
    decision. Each pass wraps its own forward under its own fresh aug; the
    conflict / all-singleton check runs after every pass and breaks the loop
    early. `deduce_mask` in `info` is the UNION (canonical frame) of all
    passes' eliminations for this round, so downstream soundness/fill
    accounting sees the full round's deductions. When `want_stats` is True,
    `info` additionally carries per-pass diagnostics:
      - `n_passes` (int): how many passes actually ran this round.
      - `per_pass_deduced` (list[int], len n_passes): bits eliminated on
        each pass (cumulative-independent — the count for that pass alone).
      - `per_pass_unsound` (list[int], len n_passes): of those, how many
        killed an alive GT bit — only populated when `orig_y` is not None
        (train path). Empty list when GT is unavailable. This is the
        per-pass compounding signal for O3. Gated behind `want_stats` like
        the other counts to avoid per-pass sync overhead when off.
    Additionally, whenever multi-pass is active (`opts.deduce_passes != 1`),
    `info["per_pass_deduce_masks"]` is a list (len n_passes) of the per-pass
    CANONICAL deduce masks (each [B, S, C], same frame/semantics as
    `info["deduce_mask"]`). Unlike the counts above these are TENSORS (no
    `.item()` sync), so they are populated regardless of `want_stats` — this
    lets the eval-time `solve()` (which runs `want_stats=False`) compute the
    per-pass-index unsound rate itself against its own ground_truth, exactly
    as it already does for the round-level `deduce_mask`. Empty at the default
    single pass.
    At the default `deduce_passes==1` the loop runs exactly once and every
    tensor / count is byte-identical to the pre-multi-pass code path.

    Decision policy (, `opts.cell_policy` /
    `opts.digit_policy`): default ("uniform"/"softmax") is byte-identical to the
    legacy uniform-cell + softmax-sample decide. `decide_rank` ([B] long) is
    only read when `digit_policy=="rank_k"` (the chain's rank within its slot,
    e.g. `row_index % K`); None -> rank 0 (== argmax). See `StepConfig`.
    """
    B, S, C = state.shape
    device = state.device
    vd = cfg.vocab_dim if cfg.vocab_dim is not None else C

    # Pre-build the dihedral cell perms once (aug frame changes per pass, but
    # the cell-perm TABLE is aug-independent). None when aug/dihedral off.
    if cfg.augment and cfg.augment_dihedral:
        n_grid = int(round(S ** 0.5))
        assert n_grid * n_grid == S, (
            f"augment_dihedral=True but S={S} is not a perfect square"
        )
        _cell_perms_table = _get_cell_perms(n_grid, device)
    else:
        _cell_perms_table = None

    def _sample_pass_aug():
        """Sample fresh per-row aug (or identity) and map inputs to that frame.

        Returns (digit_perm, dih_idx, cell_perms, state_aug, gm_aug,
        orig_y_aug, ip_aug). `state` here is the CANONICAL current state
        carried across passes. At the default single pass this is the exact
        legacy aug-sampling block — same RNG call sequence, same tensors.
        """
        if cfg.augment:
            cell_perms = _cell_perms_table
            # `permute_digits=False` overrides cfg.vocab_dim with 0, yielding
            # an identity channel-perm (digit-perm becomes a no-op). maze.
            aug_vd = 0 if not cfg.permute_digits else cfg.vocab_dim
            digit_perm, dih_idx = sample_chain_augs(
                B, C, device, with_dihedral=cfg.augment_dihedral, vocab_dim=aug_vd,
            )
            state_aug = apply_aug_state(cur_state, digit_perm, dih_idx, cell_perms)
            gm_aug = apply_aug_mask(given_mask, dih_idx, cell_perms)
            orig_y_aug = (
                apply_aug_state(orig_y, digit_perm, dih_idx, cell_perms)
                if orig_y is not None else None
            )
            ip_aug = (
                apply_aug_mask(in_puzzle_mask, dih_idx, cell_perms)
                if in_puzzle_mask is not None else None
            )
        else:
            cell_perms = None
            digit_perm = None
            dih_idx = None
            state_aug = cur_state
            gm_aug = given_mask
            orig_y_aug = orig_y
            ip_aug = in_puzzle_mask
        return digit_perm, dih_idx, cell_perms, state_aug, gm_aug, orig_y_aug, ip_aug

    # Iterated (forward + threshold-eliminate) inner loop. `deduce_passes`
    # bounds it; 0 = run to fixpoint (no bit falls below threshold), capped
    # by `deduce_pass_cap`. Default 1 -> a single pass, byte-identical to the
    # legacy path (loop body runs exactly once, no early-break evaluation).
    if opts.deduce_passes == 1:
        max_passes = 1
    elif opts.deduce_passes == 0:
        max_passes = max(1, opts.deduce_pass_cap)
    else:
        max_passes = opts.deduce_passes

    # Canonical union of every pass's eliminations this round (for downstream
    # soundness / per-round-fill accounting in solve()). Lazily allocated on
    # the first pass so the single-pass path allocates nothing extra.
    deduce_mask_canonical_union: torch.Tensor | None = None
    cur_state = state  # canonical state carried across passes
    per_pass_deduced: list[int] = []
    per_pass_unsound: list[int] = []
    # Per-pass CANONICAL deduce masks (each [B, S, C], same frame/semantics as
    # info["deduce_mask"]) — the per-pass-index compounding signal for .
    # Collected whenever multi-pass is active, INDEPENDENT of want_stats: each
    # entry is a tensor (no .item() sync), so solve() can do the GT comparison
    # itself even on the want_stats=False eval hot path. Empty at the default
    # single pass.
    per_pass_deduce_masks: list[torch.Tensor] = []
    n_passes = 0

    # These hold the FINAL pass's aug-frame artifacts; the decide + invert run
    # against them exactly as in the legacy single-pass code.
    digit_perm = dih_idx = cell_perms = None
    state_aug = gm_aug = orig_y_aug = ip_aug = None
    new_state_aug = out = sm_logits = None
    empty_cell = all_singleton = cls_fires = conflict = solved = can_decide = None

    for _pass in range(max_passes):
        n_passes += 1
        (digit_perm, dih_idx, cell_perms, state_aug, gm_aug,
         orig_y_aug, ip_aug) = _sample_pass_aug()

        out = model(state_aug, use_final=True)
        bce_logits = out["bce"]
        sm_logits = out["softmax"]

        # ===== Deduction (deterministic threshold) — in aug frame =====
        probs = torch.sigmoid(bce_logits)
        deduce_mask_aug = (probs < cfg.threshold) & (state_aug > 0.5)
        deduce_mask_aug = deduce_mask_aug & ~gm_aug.unsqueeze(-1)
        if ip_aug is not None:
            # Don't deduce on out-of-puzzle cells (they're permanently zero
            # anyway, but explicit gating keeps soundness diagnostics clean).
            deduce_mask_aug = deduce_mask_aug & ip_aug.unsqueeze(-1)
        if vd < C:
            # Don't touch the auxiliary (post-vocab) channels — e.g.
            # snowflake's locked mask channel. Build an explicit vocab-only
            # mask and AND it in.
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
        can_decide = ~conflict & ~solved            # [B]

        # ----- Multi-pass bookkeeping (skipped entirely at the default) -----
        if opts.deduce_passes != 1:
            # Invert this pass's aug-frame deduce mask to canonical and OR it
            # into the running round union. new_state_aug -> canonical is the
            # carried state for the next pass.
            if cfg.augment:
                pass_deduce_canon = invert_aug_state(
                    deduce_mask_aug.float(), digit_perm, dih_idx, cell_perms,
                ) > 0.5
                cur_state = invert_aug_state(
                    new_state_aug, digit_perm, dih_idx, cell_perms,
                )
            else:
                pass_deduce_canon = deduce_mask_aug
                cur_state = new_state_aug
            if deduce_mask_canonical_union is None:
                deduce_mask_canonical_union = pass_deduce_canon.clone()
            else:
                deduce_mask_canonical_union = deduce_mask_canonical_union | pass_deduce_canon

            # Record this pass's canonical deduce mask for the per-pass-index
            # compounding signal (). Tensor-only (no sync) so it is
            # collected regardless of want_stats — solve() reads it to do the
            # GT comparison itself on the eval hot path.
            per_pass_deduce_masks.append(pass_deduce_canon)

            if want_stats:
                per_pass_deduced.append(int(pass_deduce_canon.sum().item()))
                if orig_y is not None:
                    # Unsound = killed an alive GT bit. Compute in canonical
                    # frame against the canonical orig_y (frame-invariant
                    # boolean count either way).
                    gt_alive = (orig_y > 0.5)
                    per_pass_unsound.append(
                        int((pass_deduce_canon & gt_alive).sum().item())
                    )

            # Early-exit conditions (checked after EVERY pass):
            #  - a conflict fired on any row (state collapsed / CLS), or
            #  - every row is all-singleton (nothing left to deduce), or
            #  - this pass eliminated nothing (fixpoint reached).
            #  - it's the last allowed pass.
            if _pass == max_passes - 1:
                break
            done = bool((conflict | all_singleton).all().item())
            nothing_eliminated = bool((~pass_deduce_canon).all().item())
            if done or nothing_eliminated:
                break

    # ===== Decision: pick a multi-alive cell, pin one candidate (aug frame) =====
    # Two independent axes, both default to the legacy behavior so the block is
    # byte-identical when cell_policy=="uniform" and digit_policy=="softmax":
    #   cell axis  — uniform (legacy) | mrv | min_entropy | max_entropy
    #   digit axis — softmax (legacy) | argmax | rank_k
    # . Deterministic cell policies break ties
    # randomly (multinomial over the argmin/argmax set) so they don't bias
    # toward low cell index; digit rank_k reads the per-row `decide_rank`.
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
        N_cd = cd_b_idx.shape[0]
        cd_arange = torch.arange(N_cd, device=device)

        cell_policy = opts.cell_policy
        if cell_policy == "uniform":
            # LEGACY PATH — byte-identical to the pre-E2 code (same RNG call).
            cell_idx = torch.multinomial(cd_multi_alive, 1).squeeze(-1)
        else:
            # Deterministic policies compute a per-cell score, restrict it to
            # multi-alive cells, take the argmin/argmax SET, then break ties by
            # a uniform multinomial over that set. `multi` is the boolean pool.
            multi = cd_multi_alive > 0.5                                # [N_cd, S]
            if cell_policy == "mrv":
                # Fewest alive vocab candidates. Non-multi cells are excluded
                # by masking their score to +inf (never the min).
                alive_counts = cd_state[..., :vd].sum(dim=-1)           # [N_cd, S]
                score = torch.where(
                    multi, alive_counts,
                    torch.full_like(alive_counts, float("inf")),
                )
                target = score.min(dim=-1, keepdim=True).values        # [N_cd, 1]
                tie_set = multi & (score <= target + 1e-6)
            elif cell_policy in ("min_entropy", "max_entropy"):
                # Entropy of the softmax head over alive candidates only,
                # renormalized. sm_logits is [B, S, C]; restrict to the rows we
                # decide on and to the alive vocab channels at each cell.
                cd_sm = sm_logits[cd_b_idx]                             # [N_cd, S, C]
                alive_cells = cd_state > 0.5                            # [N_cd, S, C]
                if vd < C:
                    vmask = torch.zeros(C, dtype=torch.bool, device=device)
                    vmask[:vd] = True
                    alive_cells = alive_cells & vmask
                masked_sm = cd_sm.masked_fill(~alive_cells, float("-inf"))
                logp = torch.log_softmax(masked_sm, dim=-1)            # [N_cd, S, C]
                p = logp.exp()
                # H = -sum p*logp over alive channels; dead channels contribute
                # 0 (p==0). nan_to_num guards the -inf*0 at fully-dead cells.
                ent = -(p * logp).nan_to_num(0.0).sum(dim=-1)          # [N_cd, S]
                if cell_policy == "min_entropy":
                    score = torch.where(
                        multi, ent, torch.full_like(ent, float("inf")))
                    target = score.min(dim=-1, keepdim=True).values
                    tie_set = multi & (score <= target + 1e-6)
                else:  # max_entropy
                    score = torch.where(
                        multi, ent, torch.full_like(ent, float("-inf")))
                    target = score.max(dim=-1, keepdim=True).values
                    tie_set = multi & (score >= target - 1e-6)
            else:
                raise ValueError(f"unknown cell_policy {cell_policy!r}")
            # Random tie-break over the winning set (never empty: every
            # can_decide row has >=1 multi-alive cell by construction).
            cell_idx = torch.multinomial(tie_set.float(), 1).squeeze(-1)

        sm_at_cell = sm_logits[cd_b_idx, cell_idx]                   # [N_cd, C]
        alive_at_cell = cd_state[cd_arange, cell_idx] > 0.5          # [N_cd, C]
        sm_at_cell = sm_at_cell.masked_fill(~alive_at_cell, float("-inf"))
        if vd < C:
            # Don't ever pin to a non-vocab channel.
            sm_at_cell[..., vd:] = float("-inf")

        digit_policy = opts.digit_policy
        if digit_policy == "softmax":
            # LEGACY PATH — byte-identical to the pre-E2 code.
            if cfg.temp_decide > 0:
                sm_probs = torch.softmax(sm_at_cell / cfg.temp_decide, dim=-1)
                digit = torch.multinomial(sm_probs, 1).squeeze(-1)
            else:
                digit = sm_at_cell.argmax(dim=-1)
        elif digit_policy == "argmax":
            digit = sm_at_cell.argmax(dim=-1)
        elif digit_policy == "rank_k":
            # Chain k in a slot takes the k-th best alive digit by logit. The
            # per-row rank comes from `decide_rank` ([B] long, e.g. row%K); None
            # -> rank 0 (== argmax). Clamp rank to (alive_count-1) so we never
            # index past the alive candidates at this cell.
            if decide_rank is None:
                rank = torch.zeros(N_cd, dtype=torch.long, device=device)
            else:
                rank = decide_rank[cd_b_idx].long()
            alive_count = alive_at_cell.sum(dim=-1).clamp(min=1)        # [N_cd]
            rank = rank.clamp(min=0)
            rank = torch.minimum(rank, alive_count - 1)
            # Sort logits descending; -inf (dead) sink to the end, so index
            # `rank` (< alive_count) always lands on an alive candidate.
            order = sm_at_cell.argsort(dim=-1, descending=True)         # [N_cd, C]
            digit = order[cd_arange, rank]
        else:
            raise ValueError(f"unknown digit_policy {digit_policy!r}")

        # Zero only the vocab channels — auxiliary channels (e.g. snowflake's
        # locked mask channel) must not be touched by the decision step.
        if vd < C:
            new_state_aug[cd_b_idx, cell_idx, :vd] = 0.0
        else:
            new_state_aug[cd_b_idx, cell_idx] = 0.0
        new_state_aug[cd_b_idx, cell_idx, digit] = 1.0

    # ===== Invert aug on new_state and deduce_mask before returning =====
    # `new_state` = the FINAL pass's post-decide aug state, inverted to
    # canonical (decide only ever runs on the final pass's frame).
    if cfg.augment:
        new_state = invert_aug_state(new_state_aug, digit_perm, dih_idx, cell_perms)
        # invert_aug_state works on float; cast deduce_mask round-trip.
        final_pass_deduce_canonical = invert_aug_state(
            deduce_mask_aug.float(), digit_perm, dih_idx, cell_perms,
        ) > 0.5
    else:
        new_state = new_state_aug
        final_pass_deduce_canonical = deduce_mask_aug

    # `deduce_mask` returned to the caller is the canonical UNION of every
    # pass's eliminations this round (so solve()'s per-round soundness / fill
    # accounting sees all of them). The multi-pass loop already OR'd the final
    # pass's mask into `deduce_mask_canonical_union`, so it is complete. At the
    # default single pass the union was never built (None) -> use the final
    # pass's canonical mask, which IS the legacy `deduce_mask` exactly
    # (byte-identical).
    if deduce_mask_canonical_union is None:
        deduce_mask = final_pass_deduce_canonical
    else:
        deduce_mask = deduce_mask_canonical_union

    if want_stats:
        info = {
            "n_deduced": int(deduce_mask.sum().item()),
            "n_decided": int(can_decide.sum().item()),
            "n_conflict": int(conflict.sum().item()),
            "n_conflict_empty": int(empty_cell.sum().item()),
            "n_conflict_cls": int(cls_fires.sum().item()),
            "n_solved": int(solved.sum().item()),
            "deduce_mask": deduce_mask,
            # Per-pass diagnostics ( compounding analysis). At the default
            # single pass, `per_pass_deduced == [n_deduced]` and
            # `per_pass_unsound` is empty (eval never passes orig_y). For
            # deduce_passes != 1 these were populated inside the inner loop.
            "n_passes": n_passes,
            "per_pass_deduced": (
                per_pass_deduced if opts.deduce_passes != 1
                else [int(deduce_mask.sum().item())]
            ),
            "per_pass_unsound": per_pass_unsound,
            # Per-pass CANONICAL deduce masks (list of [B, S, C]) — tensors,
            # no sync. Present only for multi-pass (); empty at default.
            "per_pass_deduce_masks": per_pass_deduce_masks,
        }
    else:
        # Skip the .item() calls entirely. `deduce_mask` is a tensor and
        # free to include; `solve()` reads it for soundness diagnostics.
        # `n_passes` is a cheap python int (no sync), kept so callers can
        # account forwards/round without a want_stats round.
        # `per_pass_deduce_masks` is tensor-only (no sync) and lets solve()
        # compute per-pass unsound rate on the eval hot path (empty at the
        # default single pass, so byte-identical there).
        info = {
            "deduce_mask": deduce_mask,
            "n_passes": n_passes,
            "per_pass_deduce_masks": per_pass_deduce_masks,
        }
    # Always expose aug params and aug-frame inputs alongside info — the
    # trainer's grad forward typically uses `aug_forward()` directly to
    # get its own aug, but exposing the no-grad aug here keeps the
    # contract honest if a caller wants to inspect it.
    info["digit_perm"] = digit_perm
    info["dih_idx"] = dih_idx
    info["aug_state"] = state_aug
    info["aug_given_mask"] = gm_aug
    info["aug_orig_y"] = orig_y_aug
    info["aug_in_puzzle_mask"] = ip_aug
    return new_state, conflict, solved, out, info
