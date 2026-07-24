"""Streaming-queue solver for sudoku.

Maintains a fixed-size active batch of `M = batch_size // n_chains` slots.
Each slot holds one puzzle's `K = n_chains` parallel stochastic chains and
carries its own round counter. The full B = M*K-row batch is forwarded
every iteration; per-slot bookkeeping then decides what happens to each
slot independently.

Slot lifecycle (per iteration of the main forward loop):
  - Forward `dpll_step` on the whole batch (one model call).
  - Freeze rows belonging to wrong-singleton chains and to empty slots
    (those chain rows don't update; we ignore their per-step outputs).
  - For each *active* slot:
      * If any chain just solved correctly  → puzzle accepted, slot evicted.
      * Else, mark wrong-singleton chains as done (frozen until eviction).
      * Reset conflict chains (in this slot only) to that puzzle's original.
      * Increment the slot's round counter.
      * If the slot's round counter ≥ max_rounds, OR all chains are done
        with no correct solve, the puzzle times out and the slot is evicted.
  - Refill every evicted slot with the next puzzle from the queue.
  - Loop until the queue is empty AND no slot is still active.

So there is no global for loop with a single max_rounds — each puzzle gets
its own max_rounds budget, starting fresh when its slot is filled.

Augmentation is handled entirely inside `dpll_step` (see dpll.py).
This file operates strictly in the canonical (original-puzzle) frame —
state, given_mask, ground_truth and `info["deduce_mask"]` are all
canonical, so the soundness/CLS diagnostics and the post-hoc
correctness check are unchanged from the pre-aug code path.

The toggle lives at `cfg.step.augment` (StepConfig.augment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from experiments.sudoku.dpll import StepConfig, dpll_step


@dataclass
class SolveConfig:
    step: StepConfig = None
    max_rounds: int = 1000      # PER-PUZZLE round budget (not global)
    n_chains: int = 64          # chains per puzzle (slightly above HP's 48)
    batch_size: int = 512       # forward batch size; mixes M = batch_size//n_chains puzzles

    # Optional per-puzzle "if K=1 sequential" cost estimate.
    # When ON, each chain reset is recorded as an "attempt"; on the first
    # winning solve, the slot enters drain mode and continues running until
    # all chains whose current attempt has index < winning attempt's index
    # have ended (reset OR solved). Then per puzzle:
    #     forwards_seq = (winning_index + 1) * avg(attempt_duration)
    # This is slower than the upper-bound K*(round_solved+1) — extra rounds
    # in drain phase + per-chain bookkeeping. Off by default.
    estimate_sequential: bool = False
    seq_drain_max_rounds: int = 200  # cap on extra rounds spent draining

    # ----- Eval early-abort (heavy-timeout configs) -----
    # When set, once this many puzzles have TIMED OUT, stop FILLING new puzzles
    # into slots. In-flight slots are drained to completion so the returned
    # results form a contiguous prefix (see `dispatched_hi` in SolveResult and
    # the clean-prefix handling in the callers) — no fast puzzle overtakes a
    # slow one and skews the pass rate upward. None = never abort (full eval).
    eval_max_timeouts: int | None = None

    # ----- Resume support -----
    # Puzzle indices already evaluated in a previous (interrupted) run. These
    # are never filled into a slot; their outcome is carried by the caller from
    # the persisted per-puzzle progress log. Lets an interrupted eval resume
    # without re-solving finished puzzles. None/empty = evaluate all.
    already_done: set | None = None

    # ----- Streaming per-puzzle progress -----
    # Called once per puzzle at slot eviction with a dict of that puzzle's
    # outcome (idx, correct, wrong, timeout, round_solved, puzzle_calls). The
    # caller persists it (e.g. one JSONL line) so an interrupted eval can
    # resume from the last recorded puzzle. None = no callback.
    on_puzzle_done: object = None

    # Optional per-puzzle per-round trajectory of the WINNING chain.
    # When True, for each correctly-solved puzzle, record TWO related metrics:
    #
    # Cell-fills (singleton-crossings): how many cells crossed the
    # singleton boundary this round (i.e. went from multi-alive to
    # exactly-1-alive). Useful for "how many cells got committed".
    #   - deduction_fills_per_round[r]
    #   - decision_fills_per_round[r]
    #
    # Bitflips (alive bits killed): the model's actual deductive output.
    # A cell going 9-alive → 5-alive contributes 4 deduction bitflips
    # but 0 fills. A decision pin on a multi-alive cell contributes
    # (alive_count − 1) decision bitflips (kills every digit at that
    # cell except the pinned one).
    #   - deduction_bitflips_per_round[r] = deduce_mask[r].sum() over
    #     the (cell, channel) dims for the winning chain.
    #   - decision_bitflips_per_round[r] = (alive bits killed by the
    #     decide step), i.e. (post_deduce_alive − post_decide_alive).
    #     Zero for rounds where decide didn't fire.
    #
    # Length of each list = round_solved + 1 (rounds 0..round_solved
    # inclusive, the last one being the winning round). Off by default
    # so existing eval pipelines are unchanged.
    log_per_round_fill: bool = False

    # ----- E2 backtracking policy (search-process ablation) --------------
    # On a chain conflict, which state to reset it to. Default "root" is
    # behaviorally identical to the legacy reset-to-original-puzzle path (and
    # is implemented WITHOUT allocating the snapshot stack at all — see solve()).
    #   "root"          — reset to the original puzzle (baseline).
    #   "last"          — snapshot taken before the most recent decision.
    #   "geometric"     — snapshot j decisions up, j ~ Geometric(geometric_p);
    #                     j >= current depth -> root.
    #   "uniform_depth" — snapshot at a uniformly-random earlier decision.
    #   "last+negate"   — like "last" but additionally kill the pinned candidate
    #                     in the restored state (depth-1 clause learning). If the
    #                     restored cell goes empty, escalate to root.
    backtrack: str = "root"
    geometric_p: float = 0.5          # p for backtrack="geometric"
    # Kept as an explicit flag for symmetry with the README; the "last+negate"
    # policy already implies negation, but callers may set backtrack="last" and
    # learn_negation=True to the same effect. When backtrack=="last+negate" the
    # negation is on regardless.
    learn_negation: bool = False
    # Max per-chain decision depth the snapshot stack records. Overflow (a chain
    # branching deeper than this) falls back to root on the next conflict. 64 is
    # ample for Sudoku-Extreme (empirically ~<40 decisions per chain).
    snapshot_max_depth: int = 64

    def __post_init__(self):
        if self.step is None:
            self.step = StepConfig()

    def uses_snapshots(self) -> bool:
        """True iff the backtrack policy needs the per-chain snapshot stack.

        `root` (default) never touches the stack, so the default solve path is
        allocation-free and byte-identical to the pre-E2 reset-to-root code.
        """
        return self.backtrack != "root"


@dataclass
class SolveResult:
    solved: torch.Tensor       # [P] bool — the model's own accept signal
    correct: torch.Tensor      # [P] bool — solved AND matches GT (post-hoc, reporting only)
    wrong: torch.Tensor        # [P] bool — solved AND mismatches GT (reporting only)
    timeouts: torch.Tensor     # [P] bool — never produced an accept
    n_resets: torch.Tensor     # [P] long — total chain resets for this puzzle
    round_solved: torch.Tensor # [P] long — round at which winning chain solved (-1 if never)
    model_calls: int           # total forward passes (one per main loop iter)
    solution: torch.Tensor     # [P, S, C]
    n_chains: int
    # ----- Diagnostics aggregated over all (active) chain-rounds -----
    diag_total_deduced: int            # # bits removed by deduction (eligible to be GT-killing)
    diag_total_unsound_deductions: int # of those, killed an actually-correct GT bit
    diag_conflict_tp: int              # detected_conflict & (GT-bit-killed-after-deduce)
    diag_conflict_fp: int              # detected_conflict & ~(GT-killed)
    diag_conflict_fn: int              # ~detected_conflict & (GT-killed)
    diag_conflict_tn: int              # ~detected_conflict & ~(GT-killed)
    diag_active_chain_rounds: int      # denominator: total active chain-rounds processed
    # ----- E3-O3 per-pass-index compounding (multi-pass deduce only). -----
    # Indexed by deduce pass number (0..max_passes-1 seen across all rounds).
    # `per_pass_deduced_total[i]` = # bits eliminated on pass i, summed over
    # every active chain-round; `per_pass_unsound_total[i]` = of those, how
    # many killed a GT-alive bit (same GT criterion as the round-level unsound
    # diagnostic). Empty lists at the default single-pass eval (deduce_passes
    # == 1), so single-pass reports are unchanged.
    per_pass_deduced_total: list[int]
    per_pass_unsound_total: list[int]
    # ----- Optional per-puzzle "if K=1 sequential" cost (only filled if
    # cfg.estimate_sequential=True; -1 elsewhere). -----
    forwards_seq: torch.Tensor         # [P] long — (W+1) * avg_attempt_duration, or -1
    seq_winning_idx: torch.Tensor      # [P] long — winning attempt index W, or -1
    seq_attempts_done: torch.Tensor    # [P] long — # of completed attempts averaged into the metric
    # ----- Optional per-puzzle per-round trajectory (only filled if
    # cfg.log_per_round_fill=True; empty list elsewhere). Indexed by puzzle
    # idx; entry is None for puzzles that never solved correctly. -----
    deduction_fills_per_round: list[list[int] | None]      # cell-singletons
    decision_fills_per_round: list[list[int] | None]
    deduction_bitflips_per_round: list[list[int] | None]   # alive-bits killed
    decision_bitflips_per_round: list[list[int] | None]
    n_givens: list[int | None]
    # ----- Per-puzzle inference cost (always populated). -----
    puzzle_calls: torch.Tensor   # [P] long — number of model_calls between
                                 # this puzzle's slot-fill and slot-eviction.
                                 # Approximates per-puzzle inference cost in
                                 # the streaming-queue solver. -1 if puzzle
                                 # was never filled (only possible if P > Q
                                 # and we exit before all queued).
    # ----- Early-abort bookkeeping -----
    aborted: bool = False        # True if eval_max_timeouts was hit and we
                                 # stopped filling new puzzles.
    dispatched_hi: int = -1      # highest puzzle index ever filled into a slot.
                                 # On a full run == P-1. On an aborted run, the
                                 # caller keeps only the maximal gap-free prefix
                                 # [0..k] of evaluated puzzles (every idx<=k has
                                 # an outcome), so the reported percentage is an
                                 # unbiased prefix, not a fast-puzzle subsample.
    n_evaluated: int = 0         # # puzzles with a real outcome this run
                                 # (excludes already_done and never-filled).
    # ----- E2 backtracking diagnostics (aggregated over all conflicts). -----
    # `backtrack` policy string echoed for the report. The two histogram lists
    # collect one entry PER CONFLICT event (across every chain-round), so a
    # downstream plot can bin them directly:
    #   conflict_depths     — the chain's decision depth at the moment it hit
    #                         the conflict (0 == conflict before any decision).
    #   backtrack_targets   — the depth the chain was reset TO (0 == root).
    # `n_negations` / `n_unsound_negations` are only nonzero under the
    # negate-style policies; unsound = negated candidate == GT digit (measured
    # the same way the unsound-deduction diagnostic uses GT).
    backtrack_policy: str = "root"
    conflict_depths: list[int] = field(default_factory=list)
    backtrack_targets: list[int] = field(default_factory=list)
    n_negations: int = 0
    n_unsound_negations: int = 0


def solve(model, puzzle, ground_truth, given_mask, cfg: SolveConfig, *,
          in_puzzle_mask: torch.Tensor | None = None,
          label_fn=None,
          verbose: bool = True) -> SolveResult:
    """puzzle: [P, S, C], ground_truth: [P, S, C], given_mask: [P, S].

    `in_puzzle_mask: [P, S] bool` is optional — when provided, only the
    cells where it's True participate in deduce / decide / conflict
    detection / GT-bookkeeping. Cells where it's False are treated as
    "not in this puzzle" (snowflake's covering-grid setup). Sudoku
    leaves this `None` (every cell is in-puzzle).

    `label_fn: (sol, gt) -> (is_correct: bool, label: str) | None` lets
    callers customize the per-puzzle correctness check + log label. When
    `None` (default), uses cell-by-cell argmax equality and labels
    "CORRECT" / "WRONG". Maze passes a function that does BFS path-
    validation and returns labels like "VALID" / "VALID-ALT" / "WRONG".
    The boolean it returns flows into `res.correct` / `res.wrong`; the
    string is what `verbose=True` prints per puzzle.

    `verbose=False` suppresses per-puzzle log lines (use during in-training eval).

    Augmentation is handled inside `dpll_step` via `cfg.step.augment`;
    callers see only canonical-frame inputs/outputs here.

    Backtracking (E2, `cfg.backtrack`): default "root" resets a conflicting
    chain to the original puzzle — behaviorally identical to the pre-E2 path,
    and implemented WITHOUT allocating the snapshot stack (see
    `cfg.uses_snapshots()`). Non-root policies maintain a per-chain decision
    snapshot stack (`snap_state [B, D_max, S, C]` uint8 + `snap_cell`/
    `snap_digit [B, D_max]` + `depth [B]`) and reset to an earlier snapshot on
    conflict; see `SolveConfig.backtrack`.

    `estimate_sequential` semantics under partial backtracking: an "attempt" is
    a maximal run of rounds a chain spends WITHOUT a reset-to-root. A partial
    backjump (last / geometric / uniform_depth / last+negate that lands above
    root) does NOT end the attempt — it extends the same attempt, reusing the
    prefix work. Only a reset that lands at root (depth 0) ends the attempt and
    starts a new one with a fresh attempt index. This keeps the K=1 sequential
    cost estimate honest: it charges the full round span of each root-to-root
    attempt, so partial-backtracking's amortized prefix reuse is reflected as
    longer (fewer) attempts rather than being double-counted.
    """
    P, S, C = puzzle.shape
    K = cfg.n_chains
    M = max(1, cfg.batch_size // K)
    B = M * K
    device = puzzle.device

    # Eval never passes `orig_y` to dpll_step → deduction is always
    # deterministic threshold, training-only stochastic kill is off.
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
    if in_puzzle_mask is not None:
        in_puzzle_mask_b = torch.zeros(B, S, dtype=torch.bool, device=device)
    else:
        in_puzzle_mask_b = None

    # Per-slot metadata. slot_puzzle[i] = -1 means slot i is empty.
    slot_puzzle = torch.full((M,), -1, dtype=torch.long, device=device)
    slot_round = torch.zeros(M, dtype=torch.long, device=device)
    slot_resets = torch.zeros(M, dtype=torch.long, device=device)
    slot_gt_idx = torch.zeros(M, S, dtype=torch.long, device=device)
    # chain_done covers both "wrong-singleton frozen" and "slot is empty".
    chain_done = torch.ones(B, dtype=torch.bool, device=device)

    # Per-puzzle per-round trajectory (only when log_per_round_fill=True).
    # Per-row int32 buffers indexed by [B, max_rounds] for both cell-fills
    # (singleton transitions) and bitflips (alive-bits killed). We use the
    # canonical state's vocab-channel slice (everything if step.vocab_dim is
    # None) for both metrics, matching the deduce/decide frames in dpll_step.
    log_fill = cfg.log_per_round_fill
    vd = cfg.step.vocab_dim if cfg.step.vocab_dim is not None else C
    deduction_fills_out: list[list[int] | None] = [None] * P
    decision_fills_out: list[list[int] | None] = [None] * P
    deduction_bitflips_out: list[list[int] | None] = [None] * P
    decision_bitflips_out: list[list[int] | None] = [None] * P
    n_givens_out: list[int | None] = [None] * P
    if log_fill:
        deduce_fills_buf = torch.zeros(B, cfg.max_rounds, dtype=torch.int32, device=device)
        decision_fills_buf = torch.zeros(B, cfg.max_rounds, dtype=torch.int32, device=device)
        deduce_bits_buf = torch.zeros(B, cfg.max_rounds, dtype=torch.int32, device=device)
        decision_bits_buf = torch.zeros(B, cfg.max_rounds, dtype=torch.int32, device=device)

    # Per-puzzle "if K=1 sequential" tracking (only when estimate_sequential=True).
    seq = cfg.estimate_sequential
    forwards_seq_out = torch.full((P,), -1, dtype=torch.long, device=device)
    seq_winning_idx_out = torch.full((P,), -1, dtype=torch.long, device=device)
    seq_attempts_done_out = torch.zeros(P, dtype=torch.long, device=device)
    if seq:
        # Per-chain: which "attempt index" this chain is currently running,
        # and which round it started.
        chain_attempt_idx = torch.zeros(B, dtype=torch.long, device=device)
        chain_attempt_start = torch.zeros(B, dtype=torch.long, device=device)
        # Per-slot: next attempt index to assign on reset, drain mode +
        # bookkeeping, running sum/count of completed attempt durations.
        slot_next_attempt_idx = torch.zeros(M, dtype=torch.long, device=device)
        slot_drain_mode = torch.zeros(M, dtype=torch.bool, device=device)
        slot_winning_idx = torch.full((M,), -1, dtype=torch.long, device=device)
        slot_drain_start = torch.full((M,), -1, dtype=torch.long, device=device)
        slot_attempt_dur_sum = torch.zeros(M, dtype=torch.long, device=device)
        slot_attempt_dur_count = torch.zeros(M, dtype=torch.long, device=device)

    next_puzzle = 0
    total_calls = 0
    n_correct_running = 0
    n_wrong_running = 0
    n_timeout_running = 0
    n_evaluated = 0             # puzzles given a real outcome this run
    dispatched_hi = -1         # highest puzzle idx ever filled into a slot
    aborted = False            # eval_max_timeouts hit → stop filling new puzzles

    # Resume: puzzle indices already evaluated in a prior interrupted run.
    # Never filled into a slot; the caller carries their outcomes.
    _already_done = cfg.already_done or set()

    def _advance_to_fillable(np_: int) -> int:
        """Return the next queue index >= np_ that is not already-done."""
        while np_ < P and np_ in _already_done:
            np_ += 1
        return np_
    # Per-puzzle inference cost: total_calls consumed between fill and evict
    # for that puzzle. -1 if puzzle was never filled. NB: this is calls
    # *during which the slot held the puzzle* — not strictly puzzle-private
    # since each main-loop forward processes ALL active slots in parallel,
    # but it's the natural amortized per-puzzle inference cost (and equals
    # forwards_unbatched in the K=1 / batch_size=K_per_puzzle case).
    slot_calls_start = torch.zeros(M, dtype=torch.long, device=device)
    puzzle_calls_out = torch.full((P,), -1, dtype=torch.long, device=device)

    # ----- E2 backtracking: per-chain decision snapshot stack. -----
    # Only allocated for non-root policies; the default "root" path never
    # touches this and stays byte-identical + allocation-free.
    use_snap = cfg.uses_snapshots()
    negate_on = (cfg.backtrack == "last+negate") or cfg.learn_negation
    D_max = cfg.snapshot_max_depth
    if use_snap:
        # snap_state[b, d] = the post-deduce state of chain b right BEFORE its
        # (d+1)-th decision was pinned. snap_cell/digit record what got pinned
        # at that decision (needed for negation). chain_depth[b] = # decisions
        # currently on the stack for chain b (also the chain's decision depth).
        snap_state = torch.zeros(B, D_max, S, C, dtype=torch.uint8, device=device)
        snap_cell = torch.zeros(B, D_max, dtype=torch.long, device=device)
        snap_digit = torch.zeros(B, D_max, dtype=torch.long, device=device)
        chain_depth = torch.zeros(B, dtype=torch.long, device=device)
    else:
        snap_state = snap_cell = snap_digit = chain_depth = None

    # decide_rank (row % K) is passed to dpll_step for digit_policy="rank_k";
    # None (and inert) for every other digit policy. Constant per chain over its
    # life. Allocated whenever rank_k is active, independent of backtracking.
    if cfg.step.digit_policy == "rank_k":
        decide_rank = (torch.arange(B, device=device) % K)
    else:
        decide_rank = None

    # E2 backtracking diagnostics (per-conflict histograms + negation counts).
    conflict_depths: list[int] = []
    backtrack_targets: list[int] = []
    n_negations = 0
    n_unsound_negations = 0

    # Diagnostic accumulators (counted only over active chain-rows).
    diag_total_deduced = 0
    diag_total_unsound_deductions = 0
    diag_conflict_tp = 0
    diag_conflict_fp = 0
    diag_conflict_fn = 0
    diag_conflict_tn = 0
    diag_active_chain_rounds = 0

    # E3-O3 per-pass-index accumulators. Grown lazily to the max pass count
    # seen (multi-pass only; stays empty at deduce_passes==1). Each entry
    # aggregates over active chain-rounds, mirroring the round-level unsound
    # diagnostic but split by deduce pass index.
    per_pass_deduced_total: list[int] = []
    per_pass_unsound_total: list[int] = []

    # Per-puzzle label resolver: compares predicted solution to GT and
    # returns (is_correct: bool, label_str: str). Default: argmax-eq match
    # → "CORRECT" / "WRONG". Maze passes a custom label_fn (see solve()
    # docstring) that does BFS path validation.
    def _label(sol_state, gt_state) -> tuple[bool, str]:
        if label_fn is not None:
            return label_fn(sol_state, gt_state)
        sol_idx = sol_state.argmax(dim=-1)
        gt_idx = gt_state.argmax(dim=-1)
        is_c = bool((sol_idx == gt_idx).all().item())
        return is_c, ("CORRECT" if is_c else "WRONG")

    # Per-row GT digit (broadcasted from slot to chain rows lazily).
    def _gt_digits_b():
        # [B, S]: GT digit per cell, broadcast from slot to its K rows.
        return slot_gt_idx.repeat_interleave(K, dim=0)

    if verbose:
        print(f"  {'puzzle':>7} | {'outcome':>13} | {'rounds':>6} | {'resets':>7} | "
              f"{'calls':>8} || {'cor/wr/to':>10}", flush=True)

    def fill(slot: int, p: int) -> None:
        nonlocal state, original, given_mask_b, in_puzzle_mask_b
        slot_puzzle[slot] = p
        slot_round[slot] = 0
        slot_resets[slot] = 0
        slot_gt_idx[slot] = ground_truth[p].argmax(dim=-1)
        slot_calls_start[slot] = total_calls
        rows = slice(slot * K, (slot + 1) * K)
        puz = puzzle[p].unsqueeze(0).expand(K, -1, -1)
        state[rows] = puz
        original[rows] = puz
        given_mask_b[rows] = given_mask[p].unsqueeze(0).expand(K, -1)
        if in_puzzle_mask_b is not None:
            in_puzzle_mask_b[rows] = in_puzzle_mask[p].unsqueeze(0).expand(K, -1)
        chain_done[rows] = False
        if use_snap:
            # Fresh puzzle in this slot -> empty every chain's decision stack.
            chain_depth[rows] = 0
        if log_fill:
            # Reset this slot's per-round buffers — we re-use buffers across
            # puzzle generations and indexing is by slot_round which restarts.
            deduce_fills_buf[rows] = 0
            decision_fills_buf[rows] = 0
            deduce_bits_buf[rows] = 0
            decision_bits_buf[rows] = 0
            # Givens count is the # singleton cells in the input puzzle.
            n_givens_out[p] = int(given_mask[p].sum().item())
        if seq:
            # K initial attempts: chain k holds attempt idx k. Counter
            # advances to K (next reset gets idx K).
            chain_attempt_idx[rows] = torch.arange(K, device=device)
            chain_attempt_start[rows] = 0
            slot_next_attempt_idx[slot] = K
            slot_drain_mode[slot] = False
            slot_winning_idx[slot] = -1
            slot_drain_start[slot] = -1
            slot_attempt_dur_sum[slot] = 0
            slot_attempt_dur_count[slot] = 0

    def evict(slot: int, p: int) -> None:
        nonlocal n_evaluated
        n_resets_out[p] = slot_resets[slot]
        puzzle_calls_out[p] = total_calls - slot_calls_start[slot]
        slot_puzzle[slot] = -1
        chain_done[slot * K:(slot + 1) * K] = True  # freeze rows so forward is benign
        n_evaluated += 1
        # Stream this puzzle's final outcome to the caller (resume/progress log).
        if cfg.on_puzzle_done is not None:
            cfg.on_puzzle_done({
                "idx": int(p),
                "correct": bool(correct_out[p].item()),
                "wrong": bool(wrong_out[p].item()),
                "timeout": bool((~solved_out[p]).item()),
                "round_solved": int(round_solved_out[p].item()),
                "n_resets": int(n_resets_out[p].item()),
                "puzzle_calls": int(puzzle_calls_out[p].item()),
            })

    # Initial fill (skipping any already-done resume indices).
    next_puzzle = _advance_to_fillable(next_puzzle)
    for slot in range(M):
        if next_puzzle >= P:
            break
        fill(slot, next_puzzle)
        dispatched_hi = max(dispatched_hi, next_puzzle)
        next_puzzle = _advance_to_fillable(next_puzzle + 1)

    while (slot_puzzle >= 0).any():
        # `dpll_step` handles augmentation internally if cfg.step.augment;
        # `new_state` and `info["deduce_mask"]` come back in canonical frame.
        new_state, conflict, just_solved_chain, _, info = dpll_step(
            model, state, given_mask_b, cfg.step,
            in_puzzle_mask=in_puzzle_mask_b, want_stats=False,
            decide_rank=decide_rank,
        )
        # One `dpll_step` performs `info["n_passes"]` model forwards this
        # round (E3-O3 iterated deduction; == 1 at the default single pass, so
        # accounting stays byte-identical). Charge every forward to the cost
        # so `model_calls` / `puzzle_calls` remain honest forwards-per-solve.
        total_calls += int(info["n_passes"])

        # ----- Diagnostics on this round (active rows only) -----
        active_rows = ~chain_done                                              # [B]
        if active_rows.any():
            deduce_mask = info["deduce_mask"]                                  # [B, S, C] canonical
            gt_digits = _gt_digits_b()                                         # [B, S]
            # GT one-hot at the per-cell GT digit. Cells with sum>1 (multi-alive)
            # would normally have the GT bit alive; cells given as singletons are
            # protected from deduction by `given_mask` so deduce_mask there is False.
            gt_one_hot = torch.zeros_like(state, dtype=torch.bool)
            gt_one_hot.scatter_(-1, gt_digits.unsqueeze(-1), True)             # [B, S, C]

            # Was the bit alive pre-deduce AND is it the GT bit for that cell?
            bit_was_gt_alive = (state > 0.5) & gt_one_hot                      # [B, S, C]
            unsound_per_bit = deduce_mask & bit_was_gt_alive                   # [B, S, C]

            # Per-row deduction counts (active rows only).
            row_deduced = deduce_mask.sum(dim=(1, 2))                          # [B]
            row_unsound = unsound_per_bit.sum(dim=(1, 2))                      # [B]
            diag_total_deduced += int((row_deduced * active_rows).sum().item())
            diag_total_unsound_deductions += int((row_unsound * active_rows).sum().item())

            # ----- E3-O3 per-pass-index compounding (multi-pass only) -----
            # dpll_step returns one canonical deduce mask per pass (tensor,
            # no sync). For each, count active-row deduced/unsound bits the
            # SAME way as the round-level aggregate above (unsound bit =
            # deduced bit that was a GT-alive bit). Empty list at the default
            # single pass -> this whole block is skipped, keeping the
            # deduce_passes==1 path byte-identical.
            per_pass_masks = info.get("per_pass_deduce_masks") or []
            for pass_i, pass_mask in enumerate(per_pass_masks):
                pass_deduced = pass_mask.sum(dim=(1, 2))                    # [B]
                pass_unsound = (pass_mask & bit_was_gt_alive).sum(dim=(1, 2))  # [B]
                d = int((pass_deduced * active_rows).sum().item())
                u = int((pass_unsound * active_rows).sum().item())
                if pass_i >= len(per_pass_deduced_total):
                    per_pass_deduced_total.append(0)
                    per_pass_unsound_total.append(0)
                per_pass_deduced_total[pass_i] += d
                per_pass_unsound_total[pass_i] += u

            # GT-conflict label: any GT bit dead in the post-deduce state
            # (i.e., it was alive pre-deduce but is no longer alive). The
            # decide step doesn't change this — decide commits to a bit that
            # is already alive at that cell.
            gt_alive_post = (state > 0.5) & gt_one_hot & ~deduce_mask          # [B, S, C]
            gt_alive_anywhere = gt_alive_post.any(dim=-1)                      # [B, S]
            if in_puzzle_mask_b is not None:
                # Out-of-puzzle cells have no GT bits; treat them as "alive"
                # so they don't falsely contribute to gt_conflict.
                row_gt_conflict = ~(gt_alive_anywhere | ~in_puzzle_mask_b).all(dim=-1)
            else:
                row_gt_conflict = ~gt_alive_anywhere.all(dim=-1)               # [B]

            # detected_conflict comes from the dpll_step (post-deduce, pre-decide).
            tp = (conflict & row_gt_conflict & active_rows).sum().item()
            fp = (conflict & ~row_gt_conflict & active_rows).sum().item()
            fn = (~conflict & row_gt_conflict & active_rows).sum().item()
            tn = (~conflict & ~row_gt_conflict & active_rows).sum().item()
            diag_conflict_tp += int(tp); diag_conflict_fp += int(fp)
            diag_conflict_fn += int(fn); diag_conflict_tn += int(tn)
            diag_active_chain_rounds += int(active_rows.sum().item())

        # ----- Per-round trajectory measurements (active rows only) -----
        # Computed pre-freeze so `new_state` here is post-decide for all
        # rows (chain_done rows aren't frozen yet but their fills/bitflips
        # will be zero anyway since they don't change). All states are in
        # canonical frame (`info["deduce_mask"]` is canonical).
        # Records BOTH cell-fills (singleton transitions) and bitflips
        # (alive bits killed) — see SolveConfig.log_per_round_fill docstring.
        if log_fill:
            deduce_mask_for_fill = info["deduce_mask"]                          # [B, S, C] canonical
            # ---- cell-fills (singleton transitions) ----
            pre_singleton = (state[..., :vd].sum(dim=-1) == 1).sum(dim=-1)       # [B]
            state_post_deduce = state.masked_fill(deduce_mask_for_fill, 0.0)
            post_deduce_singleton = (state_post_deduce[..., :vd].sum(dim=-1) == 1).sum(dim=-1)  # [B]
            post_decide_singleton = (new_state[..., :vd].sum(dim=-1) == 1).sum(dim=-1)          # [B]
            row_deduce_fills = (post_deduce_singleton - pre_singleton).to(torch.int32)
            row_decision_fills = (post_decide_singleton - post_deduce_singleton).to(torch.int32)
            # ---- bitflips (alive bits killed) ----
            # Deduction bitflips = # of True positions in deduce_mask (each
            # corresponds to a previously-alive bit that got killed). Restrict
            # to vocab channels — auxiliary channels never participate.
            row_deduce_bits = deduce_mask_for_fill[..., :vd].sum(dim=(1, 2)).to(torch.int32)  # [B]
            # Decision bitflips = (alive bits before decide) − (alive bits
            # after decide). Decide only modifies one cell (multi-alive →
            # singleton), killing (k − 1) bits at that cell; for chains
            # where decide didn't fire (conflict / solved), this is 0.
            pre_alive_bits = (state[..., :vd] > 0.5).sum(dim=(1, 2))             # [B]
            post_decide_alive_bits = (new_state[..., :vd] > 0.5).sum(dim=(1, 2)) # [B]
            row_decision_bits = (pre_alive_bits - row_deduce_bits - post_decide_alive_bits).to(torch.int32)
            # Index per-row write into [b, slot_round[slot_of(b)]]. Since slot_round
            # is constant within a slot's K rows, build an aligned index.
            slot_round_per_row = slot_round.repeat_interleave(K)                 # [B]
            row_idx = torch.arange(B, device=device)
            active = ~chain_done                                                # [B]
            if active.any():
                idx_b = row_idx[active]
                idx_r = slot_round_per_row[active]
                deduce_fills_buf[idx_b, idx_r] = row_deduce_fills[active]
                decision_fills_buf[idx_b, idx_r] = row_decision_fills[active]
                deduce_bits_buf[idx_b, idx_r] = row_deduce_bits[active]
                decision_bits_buf[idx_b, idx_r] = row_decision_bits[active]

        # ----- E2 snapshot push (non-root policies only) -----
        # A chain "made a decision" this round iff it neither conflicted nor
        # solved AND some multi-alive cell became a singleton via the decide
        # step. We reconstruct (cell, digit) by diffing the post-deduce state
        # against new_state: exactly one cell drops from multi-alive to a
        # singleton (dpll_step pins one cell/round). The snapshot we store is
        # the POST-DEDUCE, PRE-PIN state so a later negation can kill the pinned
        # digit soundly. Restricted to active chains with stack headroom.
        if use_snap:
            deduce_mask_s = info["deduce_mask"]                      # [B, S, C] canonical
            state_post_deduce = state.masked_fill(deduce_mask_s, 0.0)
            pre_alive = (state_post_deduce[..., :vd] > 0.5).sum(dim=-1)   # [B, S]
            post_alive = (new_state[..., :vd] > 0.5).sum(dim=-1)          # [B, S]
            # Decided cell: went multi-alive (>1) -> singleton (==1).
            decided_cell_mask = (pre_alive > 1) & (post_alive == 1)      # [B, S]
            made_decision = (
                decided_cell_mask.any(dim=-1) & ~conflict
                & ~just_solved_chain & ~chain_done
            )                                                            # [B]
            has_room = chain_depth < D_max
            push_rows = (made_decision & has_room).nonzero(as_tuple=True)[0]
            if push_rows.numel() > 0:
                # cell = the (single) decided cell; digit = its surviving bit
                # in new_state. argmax over the mask / vocab gives both.
                cell_of = decided_cell_mask[push_rows].float().argmax(dim=-1)  # [n]
                depth_of = chain_depth[push_rows]
                # Snapshot the post-deduce, pre-pin state (uint8).
                snap_state[push_rows, depth_of] = (
                    state_post_deduce[push_rows] > 0.5
                ).to(torch.uint8)
                snap_cell[push_rows, depth_of] = cell_of
                digit_of = new_state[push_rows, cell_of, :vd].argmax(dim=-1)
                snap_digit[push_rows, depth_of] = digit_of
                chain_depth[push_rows] = depth_of + 1
            # Chains that decided but overflowed the stack: bump depth anyway so
            # the depth-at-conflict diagnostic stays honest and overflow -> root.
            overflow_rows = (made_decision & ~has_room)
            if overflow_rows.any():
                chain_depth[overflow_rows] = chain_depth[overflow_rows] + 1

        # Freeze wrong-singleton-frozen and empty-slot rows: don't let their
        # state mutate.
        new_state = torch.where(chain_done.view(-1, 1, 1), state, new_state)

        evictions: list[int] = []
        for slot in range(M):
            p = int(slot_puzzle[slot].item())
            if p < 0:
                continue
            lo, hi = slot * K, (slot + 1) * K
            slot_solved = just_solved_chain[lo:hi] & ~chain_done[lo:hi]
            slot_conflict = conflict[lo:hi] & ~chain_done[lo:hi]

            # Helper: record per-attempt durations for chains whose
            # attempts ended this round (only used when seq is on).
            def _record_attempts(local_idx_tensor: torch.Tensor) -> None:
                if local_idx_tensor.numel() == 0:
                    return
                global_idx = local_idx_tensor + lo
                # Each ended attempt ran (slot_round - chain_start + 1) rounds.
                durs = slot_round[slot] - chain_attempt_start[global_idx] + 1
                slot_attempt_dur_sum[slot] += int(durs.sum().item())
                slot_attempt_dur_count[slot] += int(durs.numel())

            if slot_solved.any():
                # First-solve event for the puzzle: record outcome (matches
                # existing behavior — first chain to report all-singleton wins).
                # In seq mode we ALSO enter drain mode here (defer eviction)
                # so we can collect more attempt-end events from sibling chains.
                first_solve = not (seq and bool(slot_drain_mode[slot].item()))
                if first_solve:
                    k = int(slot_solved.nonzero(as_tuple=True)[0][0].item())
                    b = lo + k
                    solved_out[p] = True
                    round_solved_out[p] = slot_round[slot]
                    solutions_out[p] = new_state[b]
                    is_correct, _solve_label = _label(new_state[b], ground_truth[p])
                    correct_out[p] = is_correct
                    wrong_out[p] = not is_correct
                    if log_fill and is_correct:
                        # Winning chain's per-round trajectory, rounds
                        # 0..round_solved inclusive (the last entry is the
                        # winning round, where decision_* should be 0 because
                        # the chain was already solved post-deduce).
                        rs = int(slot_round[slot].item())
                        deduction_fills_out[p] = deduce_fills_buf[b, : rs + 1].cpu().tolist()
                        decision_fills_out[p] = decision_fills_buf[b, : rs + 1].cpu().tolist()
                        deduction_bitflips_out[p] = deduce_bits_buf[b, : rs + 1].cpu().tolist()
                        decision_bitflips_out[p] = decision_bits_buf[b, : rs + 1].cpu().tolist()
                    if is_correct:
                        n_correct_running += 1
                    else:
                        n_wrong_running += 1
                    if seq:
                        # Record the winning attempt; enter drain mode.
                        _record_attempts(slot_solved.nonzero(as_tuple=True)[0])
                        slot_winning_idx[slot] = chain_attempt_idx[b]
                        slot_drain_mode[slot] = True
                        slot_drain_start[slot] = slot_round[slot]
                        chain_done[lo:hi] = chain_done[lo:hi] | slot_solved
                    else:
                        label = _solve_label
                        rounds = int(round_solved_out[p].item())
                        resets = int(slot_resets[slot].item())
                        if verbose:
                            print(f"  {p:>7d} | {label:>13} | {rounds:>6d} | {resets:>7d} | "
                                  f"{total_calls:>8d} || "
                                  f"{n_correct_running}/{n_wrong_running}/{n_timeout_running}",
                                  flush=True)
                        evictions.append(slot)
                        continue
                else:
                    # In drain mode and another chain solved — record the
                    # attempt and freeze the chain. Don't change puzzle outcome.
                    _record_attempts(slot_solved.nonzero(as_tuple=True)[0])
                    chain_done[lo:hi] = chain_done[lo:hi] | slot_solved

            # Reset conflict chains in this slot per the backtrack policy.
            still_conflict = slot_conflict & ~chain_done[lo:hi]
            if still_conflict.any():
                local_reset = still_conflict.nonzero(as_tuple=True)[0]
                idx = local_reset + lo
                slot_resets[slot] += int(still_conflict.sum().item())

                if not use_snap:
                    # ===== LEGACY ROOT PATH — byte-identical to pre-E2. =====
                    new_state[idx] = original[idx]
                    root_reset_local = local_reset  # all resets are root resets
                else:
                    # ===== E2 partial / negating backtrack. =====
                    # For each conflicting chain, `keep` = how many of its
                    # decisions to retain (restore state to just before decision
                    # `keep+1`). keep in [0, depth-1]; keep==depth (only when
                    # depth==0, i.e. conflict before any decision) or overflow
                    # -> root (original). The restored snapshot is snap_state
                    # [keep]; negation kills the digit pinned at decision keep+1
                    # (= snap_digit[keep]). A chain lands at ROOT (attempt ends)
                    # only when depth==0 or negation empties the restored cell.
                    n_conf = int(local_reset.numel())
                    depth_at_conf = chain_depth[idx]                    # [n_conf]
                    if cfg.backtrack in ("last", "last+negate"):
                        # Undo just the most recent decision.
                        keep = (depth_at_conf - 1).clamp(min=0)
                    elif cfg.backtrack == "geometric":
                        # Undo j decisions, j ~ Geometric(p), j>=1. j>=depth ->
                        # undo everything (keep 0). geometric_ returns
                        # #failures-before-first-success in {1,2,...} for torch.
                        j = torch.empty(n_conf, device=device).geometric_(
                            cfg.geometric_p).long().clamp(min=1)
                        keep = (depth_at_conf - j).clamp(min=0)
                    elif cfg.backtrack == "uniform_depth":
                        # Keep a uniformly-random # of decisions in [0, depth-1].
                        r = torch.rand(n_conf, device=device)
                        keep = (r * depth_at_conf.float()).floor().long().clamp(min=0)
                    else:
                        raise ValueError(f"unknown backtrack {cfg.backtrack!r}")

                    # No snapshot exists when depth==0 (never decided) or the
                    # chain overflowed the stack -> those force root.
                    overflowed = depth_at_conf > D_max
                    no_decision = depth_at_conf <= 0
                    force_root = overflowed | no_decision

                    is_root = torch.zeros(n_conf, dtype=torch.bool, device=device)
                    for jj in range(n_conf):
                        b = int(idx[jj].item())
                        d_conf = int(depth_at_conf[jj].item())
                        if bool(force_root[jj].item()):
                            new_state[b] = original[b]
                            chain_depth[b] = 0
                            is_root[jj] = True
                            conflict_depths.append(d_conf)
                            backtrack_targets.append(0)
                            continue
                        k = int(keep[jj].item())
                        # Restore the pre-pin snapshot before decision k+1.
                        new_state[b] = snap_state[b, k].float()
                        # A plain (non-negating) restore to keep==0 discards the
                        # whole prefix -> counts as a root reset (attempt ends).
                        # A negating restore adds a real constraint (kills a
                        # candidate) so it EXTENDS the attempt even at k==0,
                        # UNLESS the cell empties and we escalate to true root.
                        landed_root = (k == 0) and not negate_on
                        if negate_on:
                            # Kill the candidate pinned at decision k+1.
                            nc = int(snap_cell[b, k].item())
                            nd = int(snap_digit[b, k].item())
                            gt_here = int(slot_gt_idx[slot, nc].item())
                            n_negations += 1
                            if nd == gt_here:
                                n_unsound_negations += 1
                            new_state[b, nc, nd] = 0.0
                            if float(new_state[b, nc, :vd].sum().item()) < 0.5:
                                # Restored cell empty -> escalate to root.
                                new_state[b] = original[b]
                                k = 0
                                landed_root = True
                        chain_depth[b] = k
                        is_root[jj] = landed_root
                        conflict_depths.append(d_conf)
                        backtrack_targets.append(k)
                    root_reset_local = local_reset[is_root.cpu()]

                if seq:
                    # Only ROOT resets end an attempt; partial backjumps extend
                    # the current attempt (its prefix work is reused). See the
                    # solve() docstring on estimate_sequential semantics.
                    root_idx = root_reset_local + lo
                    _record_attempts(root_reset_local)
                    n_new = int(root_reset_local.numel())
                    if n_new > 0:
                        chain_attempt_idx[root_idx] = (
                            slot_next_attempt_idx[slot]
                            + torch.arange(n_new, device=device)
                        )
                        chain_attempt_start[root_idx] = slot_round[slot] + 1
                        slot_next_attempt_idx[slot] += n_new

            slot_round[slot] += 1

            # In seq drain mode: check whether all attempts with idx <
            # winning_idx have ended (the chains holding such idx are now
            # either done OR have been reset to a higher idx).
            if seq and bool(slot_drain_mode[slot].item()):
                win_idx = int(slot_winning_idx[slot].item())
                pending = (
                    (chain_attempt_idx[lo:hi] < win_idx) & ~chain_done[lo:hi]
                ).any().item()
                drain_age = int(slot_round[slot].item()) - int(slot_drain_start[slot].item())
                if (not pending) or drain_age >= cfg.seq_drain_max_rounds:
                    cnt = int(slot_attempt_dur_count[slot].item())
                    sm = int(slot_attempt_dur_sum[slot].item())
                    if correct_out[p]:
                        avg = sm / max(cnt, 1)
                        forwards_seq_out[p] = int(round((win_idx + 1) * avg))
                    else:
                        # wrong → upper-bound, consistent with forwards_unbatched policy
                        forwards_seq_out[p] = cfg.max_rounds * K
                    seq_winning_idx_out[p] = win_idx
                    seq_attempts_done_out[p] = cnt
                    if verbose:
                        _, label = _label(solutions_out[p], ground_truth[p])
                        rounds = int(round_solved_out[p].item())
                        avg = sm / max(cnt, 1)
                        seq_val = int(forwards_seq_out[p].item())
                        print(f"  {p:>7d} | {label:>13} | {rounds:>6d} | "
                              f"{int(slot_resets[slot].item()):>7d} | "
                              f"{total_calls:>8d} || "
                              f"{n_correct_running}/{n_wrong_running}/{n_timeout_running} | "
                              f"W={win_idx} avg={avg:.1f} seq={seq_val}",
                              flush=True)
                    evictions.append(slot)
                    continue

            # Per-slot timeout: no chain accepted before round budget ran out.
            if int(slot_round[slot].item()) >= cfg.max_rounds:
                evictions.append(slot)
                if not solved_out[p]:
                    n_timeout_running += 1
                    # Early-abort: once enough puzzles have timed out, stop
                    # filling NEW puzzles. In-flight slots still drain below,
                    # so the caller gets a contiguous evaluated prefix.
                    if (cfg.eval_max_timeouts is not None
                            and n_timeout_running >= cfg.eval_max_timeouts
                            and not aborted):
                        aborted = True
                        if verbose:
                            print(f"  [eval-abort] {n_timeout_running} timeouts "
                                  f">= {cfg.eval_max_timeouts}; draining in-flight "
                                  f"slots, no new fills.", flush=True)
                resets = int(slot_resets[slot].item())
                rounds = int(slot_round[slot].item())
                if seq and forwards_seq_out[p].item() == -1:
                    # Pure timeout (never solved) → upper-bound K * max_rounds.
                    forwards_seq_out[p] = cfg.max_rounds * K
                    seq_attempts_done_out[p] = int(slot_attempt_dur_count[slot].item())
                if verbose:
                    if not solved_out[p]:
                        label = "TIMEOUT"
                    else:
                        _, label = _label(solutions_out[p], ground_truth[p])
                    print(f"  {p:>7d} | {label:>13} | {rounds:>6d} | {resets:>7d} | "
                          f"{total_calls:>8d} || "
                          f"{n_correct_running}/{n_wrong_running}/{n_timeout_running}",
                          flush=True)

        state = new_state

        # Refill evicted slots with the next queued puzzles (or mark empty).
        # When aborted, we DON'T refill — the loop then drains as in-flight
        # slots evict and empty out.
        for slot in evictions:
            p = int(slot_puzzle[slot].item())
            if p >= 0:
                evict(slot, p)
            if not aborted and next_puzzle < P:
                fill(slot, next_puzzle)
                dispatched_hi = max(dispatched_hi, next_puzzle)
                next_puzzle = _advance_to_fillable(next_puzzle + 1)

    # A puzzle is a TIMEOUT only if it was actually filled+evicted this run
    # (puzzle_calls >= 0) and never solved. Never-filled puzzles (resume-
    # skipped, or unfilled after abort) are NOT timeouts — they carry -1 in
    # puzzle_calls and must be excluded by the caller (they are not part of
    # this run's evaluated set). This keeps `~solved` from mislabeling them.
    filled_this_run = puzzle_calls_out >= 0
    timeouts = (~solved_out) & filled_this_run
    return SolveResult(
        solved=solved_out.clone(),
        correct=correct_out.clone(),
        wrong=wrong_out.clone(),
        timeouts=timeouts,
        n_resets=n_resets_out,
        round_solved=round_solved_out,
        model_calls=total_calls,
        solution=solutions_out,
        n_chains=K,
        diag_total_deduced=diag_total_deduced,
        diag_total_unsound_deductions=diag_total_unsound_deductions,
        diag_conflict_tp=diag_conflict_tp,
        diag_conflict_fp=diag_conflict_fp,
        diag_conflict_fn=diag_conflict_fn,
        diag_conflict_tn=diag_conflict_tn,
        diag_active_chain_rounds=diag_active_chain_rounds,
        per_pass_deduced_total=per_pass_deduced_total,
        per_pass_unsound_total=per_pass_unsound_total,
        forwards_seq=forwards_seq_out,
        seq_winning_idx=seq_winning_idx_out,
        seq_attempts_done=seq_attempts_done_out,
        deduction_fills_per_round=deduction_fills_out,
        decision_fills_per_round=decision_fills_out,
        deduction_bitflips_per_round=deduction_bitflips_out,
        decision_bitflips_per_round=decision_bitflips_out,
        n_givens=n_givens_out,
        puzzle_calls=puzzle_calls_out,
        aborted=aborted,
        dispatched_hi=dispatched_hi,
        n_evaluated=n_evaluated,
        backtrack_policy=cfg.backtrack,
        conflict_depths=conflict_depths,
        backtrack_targets=backtrack_targets,
        n_negations=n_negations,
        n_unsound_negations=n_unsound_negations,
    )
