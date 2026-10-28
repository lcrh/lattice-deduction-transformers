"""Streaming-queue solver for sudoku.

Maintains a fixed-size active batch of `M = batch_size // n_chains` slots.
Each slot holds one puzzle's `K = n_chains` parallel stochastic chains and
carries its own round counter. The full B = M*K-row batch is forwarded
every iteration; per-slot bookkeeping then decides what happens to each
slot independently.

Per iteration: one `dpll_step` over the whole batch. A slot whose chain
reports a solution is accepted and evicted; conflicting chains restart (or
backtrack); a slot that reaches `max_rounds` times out. Evicted slots are
refilled from the queue, so every puzzle gets its own round budget.

Augmentation happens inside `dpll_step`; everything here is in the canonical frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, get_args

import torch
import torch.nn.functional as F

from experiments.sudoku.dpll import StepConfig, dpll_step

Backtrack = Literal["root", "last", "geometric", "uniform_depth"]
BACKTRACK_POLICIES = get_args(Backtrack)


@dataclass
class SearchStats:
    """Totals over all backtracks of a solve (non-root backtrack policies)."""

    backtracks: int = 0
    sum_conflict_depth: int = 0
    sum_target_depth: int = 0
    negations: int = 0
    unsound_negations: int = 0


class DecisionStack:
    """Per-chain stack of decisions since the last restart, for backtracking.

    Stores the post-deduction state each decision was made from and the decided
    (cell, candidate), in a ring buffer that keeps the last `max_depth` decisions.
    """

    def __init__(self, B: int, S: int, C: int, max_depth: int, device):
        self.states = torch.zeros(B, max_depth, S, C, dtype=torch.uint8, device=device)
        self.cells = torch.zeros(B, max_depth, dtype=torch.long, device=device)
        self.digits = torch.zeros(B, max_depth, dtype=torch.long, device=device)
        self.depth = torch.zeros(B, dtype=torch.long, device=device)
        self.max_depth = max_depth
        self.stats = SearchStats()

    def clear(self, rows) -> None:
        self.depth[rows] = 0

    def push(self, state, deduce_mask, new_state, decided_rows, vd: int) -> None:
        """Record the decision each of `decided_rows` just made, overwriting the oldest when full."""
        post_deduce = state.masked_fill(deduce_mask, 0.0)
        pre_alive = (post_deduce[..., :vd] > 0.5).sum(dim=-1)
        post_alive = (new_state[..., :vd] > 0.5).sum(dim=-1)
        decided_cell = (pre_alive > 1) & (post_alive == 1)
        made = decided_rows & decided_cell.any(dim=-1)
        rows = made.nonzero(as_tuple=True)[0]
        if rows.numel():
            cell = decided_cell[rows].float().argmax(dim=-1)
            d = self.depth[rows] % self.max_depth
            self.states[rows, d] = (post_deduce[rows] > 0.5).to(torch.uint8)
            self.cells[rows, d] = cell
            self.digits[rows, d] = new_state[rows, cell, :vd].argmax(dim=-1)
        self.depth[made] += 1

    def backtrack(self, rows: torch.Tensor, cfg: SolveConfig, new_state, original,
                  gt_idx, vd: int) -> torch.Tensor:
        """Move each conflicting chain in `rows` back per `cfg.backtrack`, into `new_state`.

        `gt_idx` ([S], the solution) only counts unsound negations. A target
        older than the kept decisions moves to the oldest one kept. Returns True
        per row where the chain restarted from the root.
        """
        depth = self.depth[rows]
        n = rows.numel()
        if cfg.backtrack == "last":
            target = depth - 1
        elif cfg.backtrack == "geometric":
            j = torch.empty(n, device=rows.device).geometric_(cfg.geometric_p).long()
            target = depth - j.clamp(min=1)
        else:  # uniform_depth
            target = (torch.rand(n, device=rows.device) * depth.float()).floor().long()
        target = torch.maximum(target, (depth - self.max_depth).clamp(min=0))
        at_root = depth <= 0
        for i in range(n):
            b, d = int(rows[i]), int(depth[i])
            self.stats.backtracks += 1
            self.stats.sum_conflict_depth += d
            if bool(at_root[i]):
                new_state[b] = original[b]
                self.depth[b] = 0
                continue
            k = int(target[i])
            new_state[b] = self.states[b, k % self.max_depth].float()
            at_root[i] = k == 0 and not cfg.backtrack_negate
            if cfg.backtrack_negate:
                cell = int(self.cells[b, k % self.max_depth])
                digit = int(self.digits[b, k % self.max_depth])
                self.stats.negations += 1
                self.stats.unsound_negations += int(digit == int(gt_idx[cell]))
                new_state[b, cell, digit] = 0.0
                if float(new_state[b, cell, :vd].sum()) < 0.5:
                    new_state[b] = original[b]
                    k = 0
                    at_root[i] = True
            self.depth[b] = k
            self.stats.sum_target_depth += k
        return at_root


@dataclass
class SolveConfig:
    step: StepConfig = field(default_factory=StepConfig)
    max_rounds: int = 1000      # per-puzzle round budget
    n_chains: int = 64          # chains per puzzle
    batch_size: int = 512       # forward batch; holds batch_size // n_chains puzzles

    # Drained-chain estimate of the forwards a single sequential chain would
    # need: after the first solve, the slot keeps running until every earlier
    # attempt has ended, then forwards_seq = (winning attempt + 1) * mean
    # attempt length. Slower; off by default.
    estimate_sequential: bool = False
    seq_drain_max_rounds: int = 200     # cap on extra rounds spent draining

    # Where a conflicting chain restarts:
    #   root          - the original puzzle
    #   last          - the state before its most recent decision
    #   geometric     - j decisions back, j ~ Geometric(geometric_p) >= 1
    #   uniform_depth - a uniformly random earlier decision
    # With `backtrack_negate`, the candidate chosen at that decision is also
    # eliminated. Only the last `max_decision_depth` decisions are kept.
    backtrack: Backtrack = "root"
    geometric_p: float = 0.5
    # One model call per puzzle and the highest-scoring digit in every cell: no
    # elimination, branching or search, and never an abstention (a baseline).
    one_shot: bool = False
    backtrack_negate: bool = False
    max_decision_depth: int = 64

    def __post_init__(self):
        if self.backtrack not in BACKTRACK_POLICIES:
            raise ValueError(f"backtrack must be one of {BACKTRACK_POLICIES}, "
                             f"got {self.backtrack!r}")
        if self.backtrack == "root" and self.backtrack_negate:
            raise ValueError("backtrack_negate needs a non-root backtrack policy")


@dataclass
class SolveResult:
    solved: torch.Tensor        # [P] bool, the model's own accept signal
    correct: torch.Tensor       # [P] bool, accepted and equal to the ground truth
    wrong: torch.Tensor         # [P] bool, accepted and not equal to the ground truth
    timeouts: torch.Tensor      # [P] bool, never accepted
    n_resets: torch.Tensor      # [P] long, chain resets
    round_solved: torch.Tensor  # [P] long, round of the accepted chain (-1 if none)
    model_calls: int            # total forward passes
    solution: torch.Tensor      # [P, S, C]
    n_chains: int
    puzzle_calls: torch.Tensor  # [P] long, model calls while the puzzle held a slot
    puzzle_rounds: torch.Tensor  # [P] long, solver rounds while the puzzle held a slot
    # [P, 6] long over the puzzle's active chain-rounds: candidates deduced, of
    # those the solution's own (unsound), conflict head tp / fp / fn / tn.
    puzzle_diag: torch.Tensor
    diag_active_chain_rounds: int
    # [P] long, drained-chain estimate (-1 unless cfg.estimate_sequential).
    forwards_seq: torch.Tensor
    seq_winning_idx: torch.Tensor      # [P] long, winning attempt index, or -1
    seq_attempts_done: torch.Tensor    # [P] long, attempts averaged into forwards_seq
    # [P, n_passes, 2] long: deduced / unsound per deduction pass (multi-pass only).
    puzzle_pass_diag: torch.Tensor | None = None
    search: SearchStats | None = None  # non-root backtrack policies only

    def _total(self, col: int) -> int:
        return int(self.puzzle_diag[:, col].sum())

    @property
    def diag_total_deduced(self) -> int:
        return self._total(0)

    @property
    def diag_total_unsound_deductions(self) -> int:
        return self._total(1)

    @property
    def diag_conflict_tp(self) -> int:
        return self._total(2)

    @property
    def diag_conflict_fp(self) -> int:
        return self._total(3)

    @property
    def diag_conflict_fn(self) -> int:
        return self._total(4)

    @property
    def diag_conflict_tn(self) -> int:
        return self._total(5)


def solve(model, puzzle, ground_truth, given_mask, cfg: SolveConfig, *,
          in_puzzle_mask: torch.Tensor | None = None,
          is_correct: Callable[[torch.Tensor, torch.Tensor], bool] | None = None,
          verbose: bool = False) -> SolveResult:
    """Solve every puzzle, streaming them through the batch (or in one shot, see `one_shot`).

    Args:
        puzzle, ground_truth: [P, S, C].
        given_mask: [P, S] cells fixed by the puzzle.
        in_puzzle_mask: [P, S]; cells outside it take no part (Snowflake's covering grid).
        is_correct: (solution [S, C], ground truth [S, C]) -> bool for an accepted
            chain; default is exact equality (Maze accepts any shortest path).
        verbose: Print one line per finished puzzle.
    """
    if cfg.one_shot:
        return solve_one_shot(model, puzzle, ground_truth, given_mask, cfg,
                              in_puzzle_mask=in_puzzle_mask, is_correct=is_correct)
    P, S, C = puzzle.shape
    K = cfg.n_chains
    M = max(1, cfg.batch_size // K)
    B = M * K
    device = puzzle.device

    solved_out = torch.zeros(P, dtype=torch.bool, device=device)
    correct_out = torch.zeros(P, dtype=torch.bool, device=device)
    wrong_out = torch.zeros(P, dtype=torch.bool, device=device)
    round_solved_out = torch.full((P,), -1, dtype=torch.long, device=device)
    n_resets_out = torch.zeros(P, dtype=torch.long, device=device)
    solutions_out = puzzle.clone()

    # Pre-allocated batched buffers (reused as slots are refilled).
    state = torch.zeros(B, S, C, device=device)
    original = torch.zeros(B, S, C, device=device)
    given_mask_b = torch.zeros(B, S, dtype=torch.bool, device=device)
    in_puzzle_mask_b = (torch.zeros(B, S, dtype=torch.bool, device=device)
                        if in_puzzle_mask is not None else None)

    # Per-slot metadata. slot_puzzle[i] = -1 means slot i is empty.
    slot_puzzle = torch.full((M,), -1, dtype=torch.long, device=device)
    slot_round = torch.zeros(M, dtype=torch.long, device=device)
    slot_resets = torch.zeros(M, dtype=torch.long, device=device)
    slot_gt_idx = torch.zeros(M, S, dtype=torch.long, device=device)
    # chain_done covers both "wrong-singleton frozen" and "slot is empty".
    chain_done = torch.ones(B, dtype=torch.bool, device=device)

    vd = cfg.step.vocab_dim if cfg.step.vocab_dim is not None else C

    # Drained-chain estimate (cfg.estimate_sequential).
    seq = cfg.estimate_sequential
    forwards_seq_out = torch.full((P,), -1, dtype=torch.long, device=device)
    seq_winning_idx_out = torch.full((P,), -1, dtype=torch.long, device=device)
    seq_attempts_done_out = torch.zeros(P, dtype=torch.long, device=device)
    if seq:
        # Per chain: the attempt it is running and the round that attempt started.
        chain_attempt_idx = torch.zeros(B, dtype=torch.long, device=device)
        chain_attempt_start = torch.zeros(B, dtype=torch.long, device=device)
        # Per slot: next attempt index, drain state, finished-attempt lengths.
        slot_next_attempt_idx = torch.zeros(M, dtype=torch.long, device=device)
        slot_drain_mode = torch.zeros(M, dtype=torch.bool, device=device)
        slot_winning_idx = torch.full((M,), -1, dtype=torch.long, device=device)
        slot_drain_start = torch.full((M,), -1, dtype=torch.long, device=device)
        slot_attempt_dur_sum = torch.zeros(M, dtype=torch.long, device=device)
        slot_attempt_dur_count = torch.zeros(M, dtype=torch.long, device=device)

    # Per-puzzle diagnostics, accumulated per slot while the puzzle is in it.
    max_passes = cfg.step.max_passes
    slot_diag = torch.zeros(M, 6, dtype=torch.long, device=device)
    puzzle_diag_out = torch.zeros(P, 6, dtype=torch.long, device=device)
    if max_passes > 1:
        slot_pass_diag = torch.zeros(M, max_passes, 2, dtype=torch.long, device=device)
        puzzle_pass_diag_out = torch.zeros(P, max_passes, 2, dtype=torch.long, device=device)
    else:
        puzzle_pass_diag_out = None

    def _per_slot(row_counts: torch.Tensor) -> torch.Tensor:
        """[B] per-chain counts -> [M] per-slot sums."""
        return row_counts.long().view(M, K).sum(dim=1)

    stack = (DecisionStack(B, S, C, cfg.max_decision_depth, device)
             if cfg.backtrack != "root" else None)
    # Latent-carry models (model.cfg.carry) pass a latent from step to step;
    # it restarts at zero with its chain.
    model_cfg = getattr(getattr(model, "_orig_mod", model), "cfg", None)  # unwrap torch.compile
    carry_mode = getattr(model_cfg, "carry", "off")
    carry = (torch.zeros(B, S, model_cfg.dim, device=device)
             if model_cfg is not None and model_cfg.carries_state else None)
    # rank_k: chain k of each puzzle takes the k-th best candidate.
    decide_rank = (torch.arange(B, device=device) % K
                   if cfg.step.digit_policy == "rank_k" else None)

    next_puzzle = 0
    total_calls = 0
    total_rounds = 0
    slot_calls_start = torch.zeros(M, dtype=torch.long, device=device)
    slot_rounds_start = torch.zeros(M, dtype=torch.long, device=device)
    puzzle_calls_out = torch.full((P,), -1, dtype=torch.long, device=device)
    puzzle_rounds_out = torch.full((P,), -1, dtype=torch.long, device=device)
    diag_active_chain_rounds = 0

    def fill(slot: int, p: int) -> None:
        slot_puzzle[slot] = p
        slot_round[slot] = 0
        slot_resets[slot] = 0
        slot_gt_idx[slot] = ground_truth[p].argmax(dim=-1)
        slot_calls_start[slot] = total_calls
        slot_rounds_start[slot] = total_rounds
        rows = slice(slot * K, (slot + 1) * K)
        puz = puzzle[p].unsqueeze(0).expand(K, -1, -1)
        state[rows] = puz
        original[rows] = puz
        given_mask_b[rows] = given_mask[p].unsqueeze(0).expand(K, -1)
        if in_puzzle_mask_b is not None:
            in_puzzle_mask_b[rows] = in_puzzle_mask[p].unsqueeze(0).expand(K, -1)
        chain_done[rows] = False
        if carry is not None:
            carry[rows] = 0
        slot_diag[slot] = 0
        if max_passes > 1:
            slot_pass_diag[slot] = 0
        if stack is not None:
            stack.clear(rows)
        if seq:
            # Chain k starts attempt k; the next reset starts attempt K.
            chain_attempt_idx[rows] = torch.arange(K, device=device)
            chain_attempt_start[rows] = 0
            slot_next_attempt_idx[slot] = K
            slot_drain_mode[slot] = False
            slot_winning_idx[slot] = -1
            slot_drain_start[slot] = -1
            slot_attempt_dur_sum[slot] = 0
            slot_attempt_dur_count[slot] = 0

    def evict(slot: int, p: int) -> None:
        n_resets_out[p] = slot_resets[slot]
        puzzle_diag_out[p] = slot_diag[slot]
        if max_passes > 1:
            puzzle_pass_diag_out[p] = slot_pass_diag[slot]
        puzzle_calls_out[p] = total_calls - slot_calls_start[slot]
        puzzle_rounds_out[p] = total_rounds - slot_rounds_start[slot]
        if verbose:
            log_puzzle(slot, p)
        slot_puzzle[slot] = -1
        chain_done[slot * K:(slot + 1) * K] = True

    running = {"CORRECT": 0, "WRONG": 0, "TIMEOUT": 0}

    def log_puzzle(slot: int, p: int) -> None:
        solved = bool(solved_out[p].item())
        label = ("CORRECT" if bool(correct_out[p].item()) else "WRONG") if solved else "TIMEOUT"
        running[label] += 1
        rounds = int(round_solved_out[p].item()) if solved else int(slot_round[slot].item())
        line = (f"  {p:>7d} | {label:>13} | {rounds:>6d} | {int(slot_resets[slot].item()):>7d} | "
                f"{total_calls:>8d} || "
                f"{running['CORRECT']}/{running['WRONG']}/{running['TIMEOUT']}")
        if seq and solved:
            line += (f" | W={int(seq_winning_idx_out[p].item())} "
                     f"seq={int(forwards_seq_out[p].item())}")
        print(line, flush=True)

    if verbose:
        print(f"  {'puzzle':>7} | {'outcome':>13} | {'rounds':>6} | {'resets':>7} | "
              f"{'calls':>8} || {'cor/wr/to':>10}", flush=True)

    # Initial fill.
    for slot in range(min(M, P)):
        fill(slot, next_puzzle)
        next_puzzle += 1

    while (slot_puzzle >= 0).any():
        new_state, conflict, just_solved_chain, _, info = dpll_step(
            model, state, given_mask_b, cfg.step,
            in_puzzle_mask=in_puzzle_mask_b, want_stats=False, decide_rank=decide_rank,
            carry_kw=None if carry_mode == "off" else {
                "orig_x": original if model_cfg.needs_puzzle else None, "carry": carry},
        )
        total_calls += info["n_passes"]
        total_rounds += 1
        new_carry = info["carry_out"] if carry is not None else None

        # Soundness diagnostics over active rows.
        active_rows = ~chain_done                                              # [B]
        if active_rows.any():
            deduce_mask = info["deduce_mask"]                                  # [B, S, C]
            gt_one_hot = torch.zeros_like(state, dtype=torch.bool)
            gt_one_hot.scatter_(-1, slot_gt_idx.repeat_interleave(K, dim=0).unsqueeze(-1), True)
            bit_was_gt_alive = (state > 0.5) & gt_one_hot
            row_deduced = deduce_mask.sum(dim=(1, 2))                          # [B]
            row_unsound = (deduce_mask & bit_was_gt_alive).sum(dim=(1, 2))     # [B]

            # True conflict: some cell lost its solution candidate in deduction
            # (out-of-puzzle cells have none and never count).
            gt_alive_anywhere = (bit_was_gt_alive & ~deduce_mask).any(dim=-1)  # [B, S]
            if in_puzzle_mask_b is not None:
                gt_alive_anywhere = gt_alive_anywhere | ~in_puzzle_mask_b
            row_gt_conflict = ~gt_alive_anywhere.all(dim=-1)                   # [B]
            diag_active_chain_rounds += int(active_rows.sum().item())

            slot_diag += torch.stack([
                _per_slot(row_deduced * active_rows),
                _per_slot(row_unsound * active_rows),
                _per_slot(conflict & row_gt_conflict & active_rows),
                _per_slot(conflict & ~row_gt_conflict & active_rows),
                _per_slot(~conflict & row_gt_conflict & active_rows),
                _per_slot(~conflict & ~row_gt_conflict & active_rows),
            ], dim=1)
            for i, pass_mask in enumerate(info["pass_deduce_masks"]):
                slot_pass_diag[:, i, 0] += _per_slot(pass_mask.sum(dim=(1, 2)) * active_rows)
                slot_pass_diag[:, i, 1] += _per_slot(
                    (pass_mask & bit_was_gt_alive).sum(dim=(1, 2)) * active_rows)

        if stack is not None:
            stack.push(state, info["deduce_mask"], new_state,
                       ~conflict & ~just_solved_chain & ~chain_done, vd)

        # Finished chains and empty slots keep their state.
        new_state = torch.where(chain_done.view(-1, 1, 1), state, new_state)
        if new_carry is not None:
            new_carry = torch.where(chain_done.view(-1, 1, 1), carry, new_carry)

        evictions: list[int] = []
        for slot in range(M):
            p = int(slot_puzzle[slot].item())
            if p < 0:
                continue
            lo, hi = slot * K, (slot + 1) * K
            slot_solved = just_solved_chain[lo:hi] & ~chain_done[lo:hi]
            slot_conflict = conflict[lo:hi] & ~chain_done[lo:hi]

            def _record_attempts(local_idx_tensor: torch.Tensor) -> None:
                if local_idx_tensor.numel() == 0:
                    return
                global_idx = local_idx_tensor + lo
                durs = slot_round[slot] - chain_attempt_start[global_idx] + 1
                slot_attempt_dur_sum[slot] += int(durs.sum().item())
                slot_attempt_dur_count[slot] += int(durs.numel())

            if slot_solved.any():
                # The first chain to report a solution decides the outcome. With
                # seq, the slot then drains instead of being evicted.
                first_solve = not (seq and bool(slot_drain_mode[slot].item()))
                if first_solve:
                    k = int(slot_solved.nonzero(as_tuple=True)[0][0].item())
                    b = lo + k
                    solved_out[p] = True
                    round_solved_out[p] = slot_round[slot]
                    solutions_out[p] = new_state[b]
                    if is_correct is not None:
                        ok = is_correct(new_state[b], ground_truth[p])
                    else:
                        ok = bool((new_state[b].argmax(dim=-1)
                                   == ground_truth[p].argmax(dim=-1)).all().item())
                    correct_out[p] = ok
                    wrong_out[p] = not ok
                    if not seq:
                        evictions.append(slot)
                        continue
                    _record_attempts(slot_solved.nonzero(as_tuple=True)[0])
                    slot_winning_idx[slot] = chain_attempt_idx[b]
                    slot_drain_mode[slot] = True
                    slot_drain_start[slot] = slot_round[slot]
                    chain_done[lo:hi] = chain_done[lo:hi] | slot_solved
                else:
                    # Draining: another chain solved; record its attempt, keep the outcome.
                    _record_attempts(slot_solved.nonzero(as_tuple=True)[0])
                    chain_done[lo:hi] = chain_done[lo:hi] | slot_solved

            # Conflict chains restart from the original puzzle, or backtrack
            # to an earlier decision (cfg.backtrack).
            still_conflict = slot_conflict & ~chain_done[lo:hi]
            if still_conflict.any():
                local_reset = still_conflict.nonzero(as_tuple=True)[0]
                idx = local_reset + lo
                if stack is None:
                    new_state[idx] = original[idx]
                    if new_carry is not None:
                        new_carry[idx] = 0
                else:
                    at_root = stack.backtrack(idx, cfg, new_state, original,
                                              slot_gt_idx[slot], vd)
                    local_reset = local_reset[at_root]
                    idx = local_reset + lo
                slot_resets[slot] += int(still_conflict.sum().item())
                if seq and local_reset.numel():
                    # An attempt ends only when its chain restarts from the root.
                    _record_attempts(local_reset)
                    n_new = int(local_reset.numel())
                    chain_attempt_idx[idx] = (
                        slot_next_attempt_idx[slot] + torch.arange(n_new, device=device)
                    )
                    chain_attempt_start[idx] = slot_round[slot] + 1
                    slot_next_attempt_idx[slot] += n_new

            slot_round[slot] += 1

            # Draining ends once every attempt before the winning one has ended.
            if seq and bool(slot_drain_mode[slot].item()):
                win_idx = int(slot_winning_idx[slot].item())
                pending = (
                    (chain_attempt_idx[lo:hi] < win_idx) & ~chain_done[lo:hi]
                ).any().item()
                drain_age = int(slot_round[slot].item()) - int(slot_drain_start[slot].item())
                out_of_rounds = int(slot_round[slot].item()) >= cfg.max_rounds
                if (not pending) or drain_age >= cfg.seq_drain_max_rounds or out_of_rounds:
                    cnt = int(slot_attempt_dur_count[slot].item())
                    sm = int(slot_attempt_dur_sum[slot].item())
                    if correct_out[p]:
                        avg = sm / max(cnt, 1)
                        forwards_seq_out[p] = int(round((win_idx + 1) * avg))
                    else:
                        forwards_seq_out[p] = cfg.max_rounds * K    # wrong: upper bound
                    seq_winning_idx_out[p] = win_idx
                    seq_attempts_done_out[p] = cnt
                    evictions.append(slot)
                    continue

            if int(slot_round[slot].item()) >= cfg.max_rounds:
                evictions.append(slot)
                if seq and forwards_seq_out[p].item() == -1:
                    forwards_seq_out[p] = cfg.max_rounds * K    # timeout: upper bound
                    seq_attempts_done_out[p] = int(slot_attempt_dur_count[slot].item())

        state = new_state
        if new_carry is not None:
            carry = new_carry

        # Refill evicted slots with the next queued puzzles (or mark empty).
        for slot in evictions:
            p = int(slot_puzzle[slot].item())
            if p >= 0:
                evict(slot, p)
            if next_puzzle < P:
                fill(slot, next_puzzle)
                next_puzzle += 1

    return SolveResult(
        solved=solved_out,
        correct=correct_out,
        wrong=wrong_out,
        timeouts=~solved_out,
        n_resets=n_resets_out,
        round_solved=round_solved_out,
        model_calls=total_calls,
        solution=solutions_out,
        n_chains=K,
        puzzle_calls=puzzle_calls_out,
        puzzle_rounds=puzzle_rounds_out,
        puzzle_diag=puzzle_diag_out,
        diag_active_chain_rounds=diag_active_chain_rounds,
        forwards_seq=forwards_seq_out,
        seq_winning_idx=seq_winning_idx_out,
        seq_attempts_done=seq_attempts_done_out,
        puzzle_pass_diag=puzzle_pass_diag_out,
        search=None if stack is None else stack.stats,
    )


def solve_one_shot(model, puzzle, ground_truth, given_mask, cfg: SolveConfig, *,
                   in_puzzle_mask: torch.Tensor | None = None,
                   is_correct: Callable[[torch.Tensor, torch.Tensor], bool] | None = None,
                   ) -> SolveResult:
    """One model call per puzzle; every free cell takes its highest-scoring digit."""
    P, S, C = puzzle.shape
    V = ground_truth.shape[-1]
    device = puzzle.device
    solution = puzzle.clone()
    fixed = given_mask if in_puzzle_mask is None else given_mask | ~in_puzzle_mask
    n_batches = 0
    for lo in range(0, P, cfg.batch_size):
        hi = min(lo + cfg.batch_size, P)
        logits = model(puzzle[lo:hi], return_all=True)["softmax"][-1][..., :V]
        guess = F.one_hot(logits.argmax(dim=-1), V).to(puzzle.dtype)
        solution[lo:hi, :, :V] = torch.where(fixed[lo:hi].unsqueeze(-1),
                                             puzzle[lo:hi, :, :V], guess)
        n_batches += 1
    if is_correct is not None:
        correct = torch.tensor([is_correct(solution[p], ground_truth[p]) for p in range(P)],
                               device=device)
    else:
        same = solution[..., :V].argmax(dim=-1) == ground_truth.argmax(dim=-1)
        correct = (same | (~in_puzzle_mask if in_puzzle_mask is not None else False)).all(dim=-1)
    ones = torch.ones(P, dtype=torch.long, device=device)
    return SolveResult(
        solved=torch.ones(P, dtype=torch.bool, device=device), correct=correct, wrong=~correct,
        timeouts=torch.zeros(P, dtype=torch.bool, device=device), n_resets=0 * ones,
        round_solved=0 * ones, model_calls=n_batches, solution=solution, n_chains=1,
        puzzle_calls=ones, puzzle_rounds=0 * ones,
        puzzle_diag=torch.zeros(P, 6, dtype=torch.long, device=device),
        diag_active_chain_rounds=0, forwards_seq=-ones, seq_winning_idx=-ones,
        seq_attempts_done=0 * ones,
    )
