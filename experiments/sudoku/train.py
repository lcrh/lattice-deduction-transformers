"""Pool-based trainer for sudoku.

Same `dpll_step` as the solver. Per training step:
  1. Sample batch from pool: (orig_x, state, orig_y, depth).
  2. Forward → BCE + softmax + conflict heads.
  3. Compute losses against ground truth:
       - BCE target = state * orig_y                  (lattice ∩ truth)
       - Softmax CE target = orig_y.argmax(-1)         (the GT digit per cell)
       - Conflict target = gt_conflict (= "any GT bit dead in current state")
  4. Backprop + step.
  5. `dpll_step` (no-grad) advances state.
  6. Pool discard policy:
       - `solved` → discard, backfill from fresh data
       - `detected_conflict AND gt_conflict_post` → discard, backfill
       - `depth >= max_depth` → discard, backfill
       - everything else stays in pool with new state (FNs/FPs both retained
         so they keep getting training signal next iteration)

The pool carries `orig_x` so given-cell protection in `dpll_step` always
refers to the *original* puzzle's givens, not the current state's
(committed) singletons.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig, SudokuExtremeDataset
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.models.weighted_bce import weighted_bce_with_logits
from lattice_diffusion.training.utils.checkpoint import save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import (
    aug_forward,
    build_dihedral_cell_perms,
)
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.solve import SolveConfig, solve


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1   # warmup_steps = int(steps * warmup_fraction)
    seed: int = 0

    bce_pos_mult: float = 4.0
    bce_neg_mult: float = 0.5
    softmax_loss_weight: float = 0.2
    conflict_loss_weight: float = 1.0

    # Deep-supervision mode. "all" (default) accumulates the per-iteration
    # loss over every loop iteration; "final" only supervises the last
    # iteration (index n_loops-1). Both keep the same 1/n_loops normalization
    # so loss magnitudes stay comparable across modes.
    supervise: str = "all"

    # Augment toggle lives at `step.augment` (StepConfig). Both the
    # grad-tracked forward (`aug_forward`) and the no-grad
    # `dpll_step` read from there.
    step: StepConfig = field(default_factory=StepConfig)
    max_age: int = 100  # age = train steps since backfill (counts up regardless of sampling)
    compile: bool = True

    # ----- E2 S3-phase-2: pool-side backtrack-matched training -----
    # Default "root" reproduces the legacy discard-and-backfill on true-positive
    # conflict (BYTE-IDENTICAL). Non-root policies instead RESTORE the pool
    # entry to a rolled-back snapshot on true-positive conflict (the entry
    # survives with its rolled-back state; age keeps ticking so max_age still
    # bounds residency; `solved` and `max_age` still discard+backfill). This is
    # what changes the training state distribution to match an inference-time
    # partial-backtracking solver. `geometric_p` / `learn_negation` mirror
    # SolveConfig. `pool_snapshot_max_depth` sizes the pool's own snapshot stack.
    backtrack: str = "root"
    geometric_p: float = 0.5
    learn_negation: bool = False
    pool_snapshot_max_depth: int = 64

    # EMA of trainable params, evaluated at eval time. TRM paper recommends
    # decay=0.999. Shadow + live state are both saved — eval (in run.py)
    # auto-swaps EMA into the live model if `ema_state_dict` is present in
    # the checkpoint extras.
    use_ema: bool = False
    ema_decay: float = 0.999

    log_every: int = 20
    eval_every: int = 100             # in-train mini-solve frequency
    # In-train eval: same 200-puzzle set as final eval, same K=64, but a
    # very tight `max_rounds`. Reports how many puzzles solve in the
    # trivial-deduction tail. With M = batch_size/K = 8, each puzzle gets
    # ≤ eval_max_rounds forwards in its slot, total forwards ≤
    # 200/M * eval_max_rounds = 25*5 ≈ 125 forwards = ~3s wall.
    eval_n_puzzles: int = 200
    eval_max_rounds: int = 5
    eval_n_chains: int = 64

    out_dir: str = "checkpoints/sudoku"
    name: str = ""
    # When True, the checkpoint path is deterministic (`{name}.pt`, no
    # wallclock timestamp) and `train()` refuses to overwrite an existing
    # file unless `overwrite=True`. Default False preserves the legacy
    # timestamped `{name}_{ts}.pt` behavior.
    no_timestamp: bool = False
    overwrite: bool = False

    model: LoopedTransformerConfig = field(default_factory=LoopedTransformerConfig)
    data: SudokuExtremeConfig = field(default_factory=SudokuExtremeConfig)


def _gt_conflict(
    state: torch.Tensor,            # [B, S, C]
    orig_y: torch.Tensor,           # [B, S, C] (one-hot for K=1; multi-alive α for K>1)
    in_puzzle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B] bool: True iff any in-puzzle cell has no state bit consistent with orig_y.

    For one-hot orig_y (K=1): equivalent to "any GT bit dead in state", i.e.
    equivalent to:
        gt_idx = orig_y.argmax(-1)
        gt_alive = state.gather(-1, gt_idx) > 0.5
        return ~gt_alive.all(-1)
    For multi-alive α(surviving) orig_y (K>1): "any cell where state ∩ α has
    no alive bit" — i.e., committed against all surviving solutions.
    """
    consistent = (state * orig_y).any(dim=-1)  # [B, S] — channel alive in both?
    if in_puzzle_mask is not None:
        return ~(consistent | ~in_puzzle_mask).all(dim=-1)
    return ~consistent.all(dim=-1)


def _alpha_surviving(
    state: torch.Tensor,           # [B, S, C]
    solutions: torch.Tensor,       # [B, K, S, C] one-hot per solution
    last_alpha: torch.Tensor | None = None,  # [B, S, C] — pre-step α (see below)
    in_puzzle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """α(surviving K solutions) per pool entry. Returns [B, S, C] float.

    A solution k "survives" iff state's commitments at every (in-puzzle) cell
    keep at least one channel that's also alive in solution k. α(surviving)
    aggregates those solutions per (cell, channel) via OR.

    When no solution survives (model has committed against all K), we fall
    back to `last_alpha` — the α from the previous step, when |surviving| was
    non-empty (because an entry only stays in the pool while not in detected-
    conflict, so the previous step's α is well-defined). This guarantees:
      - `_gt_conflict(state, α)` still fires (state must have killed at least
        one bit of last_alpha to make |surviving| transition to 0).
      - BCE target = `state * α` carries real positive supervision at every
        cell where state didn't kill last_alpha's bit (multi-alive at branching
        cells where last_alpha was OR-of-multiple-survivors), and ∅ at the
        cells where state did kill it (negative supervision exactly on the
        conflict cells).

    If `last_alpha` is None, falls back to `solutions[:, 0]` (the canonical
    GT). For K=1 this is identical to last_alpha (which always equals
    `solutions[:, 0]` in steady state), so K=1 callers can omit it.
    """
    sol_bool = solutions > 0.5                                          # [B, K, S, C]
    state_alive = state > 0.5                                            # [B, S, C]
    consistent = (state_alive.unsqueeze(1) & sol_bool).any(dim=-1)       # [B, K, S]
    if in_puzzle_mask is not None:
        consistent = consistent | ~in_puzzle_mask.unsqueeze(1)
    surviving = consistent.all(dim=-1)                                   # [B, K]
    any_surviving = surviving.any(dim=-1)                                # [B]
    alpha = (sol_bool & surviving.unsqueeze(-1).unsqueeze(-1)).any(dim=1)  # [B, S, C]
    fallback = last_alpha if last_alpha is not None else sol_bool[:, 0].float()
    alpha = torch.where(
        any_surviving.unsqueeze(-1).unsqueeze(-1),
        alpha.float(),
        fallback,
    )
    return alpha


def _given_mask(orig_x: torch.Tensor) -> torch.Tensor:
    """[B, S] bool: cells that started as singletons in the original puzzle."""
    return (orig_x.sum(dim=-1) == 1)


def _losses(out, state, orig_y, given_mask, is_sat, gt_conflict_target,
            bce_pos_mult, bce_neg_mult, softmax_w, conflict_w,
            supervise="all"):
    B, S, C = state.shape
    device = state.device
    bce_target = state * orig_y
    pos_w = torch.full((B, 1, 1), bce_pos_mult, device=device)
    neg_w = torch.full((B, 1, 1), bce_neg_mult, device=device)

    # Cells where α has multiple alive channels — the model has multiple
    # legitimate options here, so we skip the softmax-CE supervision (which
    # would arbitrarily pick `argmax` as the target). For K=1 one-hot orig_y,
    # multi_alive_target is False everywhere → ce_mask covers all in-puzzle
    # cells.
    multi_alive_target = orig_y.sum(dim=-1) > 1                          # [B, S]

    n_loops = len(out["bce"])
    # Deep supervision: "all" sums the loss over every iteration; "final"
    # only accumulates the last iteration's loss. We normalize by the number
    # of *supervised* iterations (not n_loops) so the per-iteration loss
    # magnitude — and hence the effective gradient scale — is the same in
    # both modes. Dividing "final" by n_loops instead would train it at
    # ~1/n_loops the gradient scale, turning the D2 ablation into a hidden
    # learning-rate cut.
    if supervise == "final":
        iter_indices = (n_loops - 1,)
    else:
        iter_indices = range(n_loops)
    total = torch.zeros((), device=device)
    for i in iter_indices:
        total = total + weighted_bce_with_logits(out["bce"][i], bce_target, pos_w, neg_w)

        # Softmax CE on non-given, single-alive-target cells of SAT puzzles only.
        sm_logits = out["softmax"][i]
        gt_idx = orig_y.argmax(dim=-1)
        ce_mask = ~given_mask & ~multi_alive_target & is_sat.unsqueeze(-1)
        if ce_mask.any():
            total = total + softmax_w * F.cross_entropy(
                sm_logits[ce_mask], gt_idx[ce_mask],
            )

        if "conflict" in out and conflict_w > 0:
            c_logits = out["conflict"][i].squeeze(-1)
            total = total + conflict_w * F.binary_cross_entropy_with_logits(
                c_logits, gt_conflict_target.float(),
            )
    return total / len(iter_indices)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.no_timestamp:
        # Deterministic path used by the followup launchers so downstream
        # experiments can find checkpoints at a fixed location.
        ckpt_path = out_dir / f"{cfg.name}.pt"
        if ckpt_path.exists() and not cfg.overwrite:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint {ckpt_path}; "
                f"pass overwrite=True to replace it."
            )
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = out_dir / f"{cfg.name}_{ts}.pt"
    # In-train mini-solve curve, one JSON object per eval point, written
    # incrementally next to the checkpoint. Truncated fresh at train start so
    # a killed run still leaves partial (but valid) curve data. Only emitted
    # for followup runs (deterministic `no_timestamp` path, where collect.py
    # reads it) — plain benchmark runs keep their original side-effect-free
    # output. `None` disables the incremental writes below.
    if cfg.no_timestamp and cfg.eval_every > 0:
        curve_path = ckpt_path.with_suffix(".train_curve.jsonl")
        curve_path.write_text("")
    else:
        curve_path = None

    model = PowersetModel(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  steps={cfg.steps}  bs={cfg.batch_size}",
          flush=True)
    if cfg.compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)

    cfg.data.batch_size = cfg.batch_size
    dataset = SudokuExtremeDataset(cfg.data)

    # Snapshot a fixed batch of test-distribution SAT puzzles for in-train eval.
    # IMPORTANT: pull from the SAME 200-puzzle set the final eval uses, then
    # take the first `eval_n_puzzles` SAT puzzles. SudokuExtremeDataset's
    # subset selection is `n_puzzles`-dependent (np.random.default_rng(seed)
    # .choice(N, k) gives different subsets at different k even with the
    # same seed), so requesting n_puzzles=200 matches the final eval's
    # subset; requesting the in-train's eval_n_puzzles=16 directly would
    # land on a different (and harder) subset.
    eval_ds = SudokuExtremeDataset(SudokuExtremeConfig(
        cache_dir=cfg.data.cache_dir, split="test",
        n_puzzles=200, batch_size=200,
        seed=200,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        augment_digit_perm=False, augment_dihedral=False,
    ))
    ex_x, ex_y, ex_sat = eval_ds.next_batch()
    eval_ds.close()
    sat_mask = ex_sat.bool()
    eval_x = ex_x[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    eval_y = ex_y[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    eval_given = (eval_x.sum(dim=-1) == 1)
    print(f"In-train eval set: {eval_x.shape[0]} SAT puzzles "
          f"(every {cfg.eval_every} steps, "
          f"K={cfg.eval_n_chains}, max_rounds={cfg.eval_max_rounds})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay, betas=cfg.betas)
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_fraction))
    scheduler = make_cosine_scheduler(optimizer, cfg.steps, warmup_steps)

    ema: ModelEMA | None = None
    if cfg.use_ema:
        ema = ModelEMA(model, cfg.ema_decay)
        print(f"EMA enabled (decay={cfg.ema_decay}); shadowing "
              f"{len(ema.shadow)} param tensors", flush=True)

    def fresh_batch(n: int):
        """Return `(x, solutions)` where `solutions` has a leading K dim.

        Sudoku/snowflake have K=1 by construction; this wrapper just adds
        an extra K dim so the trainer flows through the K-aware code path
        uniformly. K>1 datasets (maze with sampled paths) populate K with
        their alternate solutions.
        """
        bx, by = [], []
        while sum(t.shape[0] for t in bx) < n:
            x_b, y_b, _ = dataset.next_batch()
            bx.append(x_b); by.append(y_b)
        x = torch.cat(bx, dim=0)[:n].to(device).float()
        y = torch.cat(by, dim=0)[:n].to(device).float()
        solutions = y.unsqueeze(1)  # [n, K=1, S, C]
        return x, solutions

    # Pool: orig_x stays fixed per entry; state evolves; `solutions` is the
    # list of K ground truths (K=1 for sudoku/snowflake). At each train step
    # we compute α(surviving) from `state` ∩ `solutions` per pool entry.
    # `last_alpha` is the α from the previous step (or, for fresh entries,
    # OR-over-all-K-solutions evaluated at the permissive initial state):
    # used as the fallback when |surviving|=0 at a step. For K=1 this is
    # always `solutions[:, 0]` in steady state, so the dynamic and static
    # GT coincide. age tracks train steps since this entry was backfilled
    # in — increments every step regardless of whether sampled, then forces
    # discard at max_age. All pool tensors are stored in CANONICAL frame.
    pool_size = cfg.batch_size
    pool_orig_x, pool_solutions = fresh_batch(pool_size)
    pool_state = pool_orig_x.clone()
    # At fresh state, all K solutions survive (orig_x is permissive, so every
    # cell is multi-alive on all GT bits). α-of-surviving = OR over K.
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P, S, C]
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  (= bs={cfg.batch_size})  "
          f"augment={cfg.step.augment}", flush=True)

    # ----- E2 S3-phase-2: pool-side decision snapshot stack. -----
    # Only maintained when a non-root backtrack policy is active; the default
    # "root" trainer never allocates it and keeps discard+backfill byte-
    # identical. Same layout as the solver's stack, indexed by pool entry.
    pool_use_snap = (cfg.backtrack != "root")
    pool_negate = (cfg.backtrack == "last+negate") or cfg.learn_negation
    D_max_pool = cfg.pool_snapshot_max_depth
    S_dim, C_dim = pool_state.shape[1], pool_state.shape[2]
    vd_pool = cfg.step.vocab_dim if cfg.step.vocab_dim is not None else C_dim
    if pool_use_snap:
        pool_snap_state = torch.zeros(
            pool_size, D_max_pool, S_dim, C_dim, dtype=torch.uint8, device=device)
        pool_snap_cell = torch.zeros(pool_size, D_max_pool, dtype=torch.long, device=device)
        pool_snap_digit = torch.zeros(pool_size, D_max_pool, dtype=torch.long, device=device)
        pool_depth = torch.zeros(pool_size, dtype=torch.long, device=device)
        print(f"E2 pool-backtrack matching: backtrack={cfg.backtrack} "
              f"negate={pool_negate} D_max={D_max_pool}", flush=True)
    else:
        pool_snap_state = pool_snap_cell = pool_snap_digit = pool_depth = None
    n_pool_restore_total = 0

    # Sudoku-specific dihedral cell permutations for spatial aug — built
    # once and reused by both `aug_forward` (grad path) and `dpll_step`
    # (no-grad path; dpll.py caches its own copy keyed by device).
    S_pool = pool_state.shape[1]
    n_grid = int(round(S_pool ** 0.5))
    if cfg.step.augment:
        assert n_grid * n_grid == S_pool, (
            f"augment=True but S={S_pool} is not a perfect square"
        )
        cell_perms_train = build_dihedral_cell_perms(n=n_grid, device=device)
    else:
        cell_perms_train = None

    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0

    # Wallclock instrumentation. step 1 includes torch.compile dispatch
    # (slow); we report it separately and time post-compile training (steps
    # 2..N) excluding in-train eval blocks.
    step1_compile_secs = 0.0
    intrain_eval_secs = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loop_start = time.perf_counter()
    t_step1_end = t_loop_start  # placeholder; overwritten after step 1

    for step in range(1, cfg.steps + 1):
        # Sample batch_size indices from the pool (without replacement within step).
        # When pool_size == batch_size this is just a permutation of the whole pool.
        sample_idx = torch.randperm(pool_size, device=device)[:cfg.batch_size]
        # All sampled tensors are in CANONICAL frame.
        state = pool_state[sample_idx]
        solutions = pool_solutions[sample_idx]   # [B, K, S, C]
        last_alpha = pool_last_alpha[sample_idx]  # [B, S, C]
        orig_x = pool_orig_x[sample_idx]
        age = pool_age[sample_idx]
        given_mask = _given_mask(orig_x)
        # `orig_y` for this step is the dynamic α(surviving K solutions),
        # falling back to `last_alpha` when |surviving|=0. For K=1 with
        # one-hot solutions, last_alpha is always `solutions[:, 0]`, so
        # this collapses to the canonical static GT.
        orig_y = _alpha_surviving(state, solutions, last_alpha)
        # gt_conflict / is_sat are bijection-invariant — compute on canonical.
        gt_conflict_pre = _gt_conflict(state, orig_y)
        is_sat_pre = ~gt_conflict_pre

        # ---- forward + losses (grad-tracked; aug applied via aug_forward) ----
        # When cfg.step.augment is True, aug_forward samples fresh
        # per-row aug and the model sees augmented state/given_mask/
        # orig_y. Loss is then computed against aug-frame targets so
        # gradients flow through the augmented operator. With
        # cfg.step.augment=False this is a strict pass-through.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(
            model, state, given_mask, orig_y=orig_y,
            augment=cfg.step.augment,
            cell_perms=cell_perms_train,
            return_all=True,
        )
        loss = _losses(
            out,
            aug_info["aug_state"], aug_info["aug_orig_y"], aug_info["aug_given_mask"],
            is_sat_pre, gt_conflict_pre,
            cfg.bce_pos_mult, cfg.bce_neg_mult,
            cfg.softmax_loss_weight, cfg.conflict_loss_weight,
            supervise=cfg.supervise,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n_examples_seen += cfg.batch_size

        # ---- step state forward (no grad) ----
        # `dpll_step` samples its own fresh aug internally (different
        # from the aug used in the grad forward above; that's fine, the
        # per-step operator is "model evaluated under random aug") and
        # returns canonical-frame `new_state` + `info["deduce_mask"]`.
        # `info` stats (n_deduced/n_decided/...) are only consumed in the
        # log_every branch below; gating them here saves 6 .item() syncs/step.
        model.eval()
        want_info_stats = (step % cfg.log_every == 0)
        with torch.no_grad():
            new_state, detected_conflict, solved, _, info = dpll_step(
                model, state, given_mask, cfg.step, orig_y=orig_y,
                want_stats=want_info_stats,
            )
        gt_conflict_post = _gt_conflict(new_state, orig_y)

        # ---- discard policy on the sampled batch ----
        new_age = age + 1
        age_exceeded = new_age > cfg.max_age
        true_positive_conflict = detected_conflict & gt_conflict_post

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())

        # ---- E2 pool-side snapshot push + backtrack-restore (non-root only) --
        # Mirrors the solver's per-chain snapshot machinery, indexed by pool
        # entry. On a true-positive conflict we RESTORE the entry to a rolled-
        # back snapshot (keeping it in the pool) instead of discarding it, so
        # the training state distribution matches an inference-time partial-
        # backtracking solver. Default backtrack="root" skips this entirely and
        # discard = solved | tp_conflict | age_exceeded (byte-identical legacy).
        restored = torch.zeros_like(true_positive_conflict)
        if pool_use_snap:
            # Gather this batch's snapshot stacks from the pool.
            b_depth = pool_depth[sample_idx].clone()                      # [B]
            b_snap_state = pool_snap_state[sample_idx]                    # [B,D,S,C]
            b_snap_cell = pool_snap_cell[sample_idx]                      # [B,D]
            b_snap_digit = pool_snap_digit[sample_idx]                   # [B,D]
            # Detect this step's decision per entry (post-deduce vs new_state).
            dmask = info["deduce_mask"]                                  # [B,S,C]
            state_post_deduce = state.masked_fill(dmask, 0.0)
            pre_alive = (state_post_deduce[..., :vd_pool] > 0.5).sum(dim=-1)  # [B,S]
            post_alive = (new_state[..., :vd_pool] > 0.5).sum(dim=-1)         # [B,S]
            decided_cell_mask = (pre_alive > 1) & (post_alive == 1)          # [B,S]
            made_decision = (
                decided_cell_mask.any(dim=-1) & ~detected_conflict & ~solved)
            has_room = b_depth < D_max_pool
            push_rows = (made_decision & has_room).nonzero(as_tuple=True)[0]
            if push_rows.numel() > 0:
                cell_of = decided_cell_mask[push_rows].float().argmax(dim=-1)
                depth_of = b_depth[push_rows]
                b_snap_state[push_rows, depth_of] = (
                    state_post_deduce[push_rows] > 0.5).to(torch.uint8)
                b_snap_cell[push_rows, depth_of] = cell_of
                b_snap_digit[push_rows, depth_of] = new_state[
                    push_rows, cell_of, :vd_pool].argmax(dim=-1)
                b_depth[push_rows] = depth_of + 1
            overflow_rows = (made_decision & ~has_room)
            if overflow_rows.any():
                b_depth[overflow_rows] = b_depth[overflow_rows] + 1

            # Restore true-positive-conflict entries to a rolled-back snapshot
            # (unless overflowed / depth 0 -> those fall through to discard).
            tp_rows = (true_positive_conflict).nonzero(as_tuple=True)[0]
            for jj in range(tp_rows.numel()):
                b = int(tp_rows[jj].item())
                d = int(b_depth[b].item())
                if d <= 0 or d > D_max_pool:
                    continue  # no snapshot -> leave for discard+backfill
                if cfg.backtrack in ("last", "last+negate"):
                    keep = d - 1
                elif cfg.backtrack == "geometric":
                    j = max(1, int(torch.empty(1, device=device).geometric_(
                        cfg.geometric_p).item()))
                    keep = max(0, d - j)
                elif cfg.backtrack == "uniform_depth":
                    keep = int(torch.rand(1, device=device).item() * d)
                else:
                    keep = d - 1
                restored_state = b_snap_state[b, keep].float()
                ok = True
                if pool_negate:
                    nc = int(b_snap_cell[b, keep].item())
                    nd = int(b_snap_digit[b, keep].item())
                    restored_state[nc, nd] = 0.0
                    if float(restored_state[nc, :vd_pool].sum().item()) < 0.5:
                        ok = False  # cell emptied -> escalate to discard
                if ok:
                    new_state = new_state.clone()
                    new_state[b] = restored_state
                    b_depth[b] = keep
                    restored[b] = True
                    n_pool_restore_total += 1
            # Scatter the (updated) snapshot stacks back to the pool.
            pool_snap_state[sample_idx] = b_snap_state
            pool_snap_cell[sample_idx] = b_snap_cell
            pool_snap_digit[sample_idx] = b_snap_digit
            pool_depth[sample_idx] = b_depth

        # Restored entries survive (NOT discarded); solved / age / non-restored
        # tp-conflicts still discard+backfill.
        discard = solved | (true_positive_conflict & ~restored) | age_exceeded

        # ---- backfill discarded sampled entries ----
        # Pool tensors are CANONICAL; new_state is canonical (dpll_step
        # inverted any aug); fresh data is canonical (the dataset has aug
        # off in our config). Straight canonical assignment.
        # `new_last_alpha` defaults to this step's `orig_y` for surviving
        # entries (= the α we just used; correct fallback for next step's
        # |surviving|=0 case). Backfilled entries get OR-over-K evaluated at
        # the fresh permissive state.
        n_to_replace = int(discard.sum().item())
        new_last_alpha = orig_y.clone()
        if n_to_replace > 0:
            fx, f_solutions = fresh_batch(n_to_replace)
            new_state = new_state.clone()
            new_orig_x = orig_x.clone()
            new_solutions = solutions.clone()
            new_age = new_age.clone()
            new_state[discard] = fx
            new_orig_x[discard] = fx
            new_solutions[discard] = f_solutions
            new_age[discard] = 0
            new_last_alpha[discard] = f_solutions.bool().any(dim=1).float()
        else:
            new_orig_x = orig_x
            new_solutions = solutions

        # Write the sampled batch's evolution back into the pool at sample_idx.
        pool_state[sample_idx] = new_state
        pool_orig_x[sample_idx] = new_orig_x
        pool_solutions[sample_idx] = new_solutions
        pool_last_alpha[sample_idx] = new_last_alpha
        pool_age[sample_idx] = new_age
        if pool_use_snap and n_to_replace > 0:
            # Backfilled (discarded) entries start with an empty decision stack.
            reset_depth = pool_depth[sample_idx]
            reset_depth[discard] = 0
            pool_depth[sample_idx] = reset_depth

        # ---- in-train eval (mini-solve on held-out SAT puzzles) ----
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            _eval_t0 = time.perf_counter()
            model.eval()
            ema_backup = ema.swap_in(model) if ema is not None else None
            with torch.no_grad():
                eval_solve_cfg = SolveConfig(
                    step=cfg.step,  # solve() doesn't pass orig_y, so deduction is deterministic;
                                    # `cfg.step.augment` controls eval-time aug
                    max_rounds=cfg.eval_max_rounds,
                    n_chains=cfg.eval_n_chains,
                    # Match training batch shape so torch.compile reuses its kernel.
                    batch_size=cfg.batch_size,
                )
                eval_res = solve(
                    model, eval_x, eval_y, eval_given, eval_solve_cfg,
                    verbose=False,
                )
            if ema is not None and ema_backup is not None:
                ema.swap_out(model, ema_backup)
            n_e = eval_res.solved.shape[0]
            n_cor_e = int(eval_res.correct.sum().item())
            n_wr_e = int(eval_res.wrong.sum().item())
            n_to_e = int(eval_res.timeouts.sum().item())
            den = max(eval_res.diag_total_deduced, 1)
            unsound_rate_e = eval_res.diag_total_unsound_deductions / den
            cls_p_e = eval_res.diag_conflict_tp / max(
                eval_res.diag_conflict_tp + eval_res.diag_conflict_fp, 1)
            cls_r_e = eval_res.diag_conflict_tp / max(
                eval_res.diag_conflict_tp + eval_res.diag_conflict_fn, 1)
            print(
                f"  [intrain-eval step={step}, max_rounds={cfg.eval_max_rounds}] "
                f"correct={n_cor_e}/{n_e}  wrong={n_wr_e}  to={n_to_e}  "
                f"calls={eval_res.model_calls}  "
                f"unsound={unsound_rate_e:.3%}  "
                f"cls:P={cls_p_e:.2f}/R={cls_r_e:.2f}",
                flush=True,
            )
            # Append this eval point to the structured train curve. Written
            # incrementally so a killed run keeps partial curve data. Skipped
            # on plain benchmark runs (curve_path is None; see train start).
            if curve_path is not None:
                with curve_path.open("a") as _cf:
                    _cf.write(json.dumps({
                        "step": step,
                        "correct": n_cor_e,
                        "wrong": n_wr_e,
                        "calls": eval_res.model_calls,
                        "unsound_rate": unsound_rate_e,
                        "cls_p": cls_p_e,
                        "cls_r": cls_r_e,
                    }) + "\n")
            if device.type == "cuda":
                torch.cuda.synchronize()
            intrain_eval_secs += time.perf_counter() - _eval_t0

        # ---- logging ----
        if step % cfg.log_every == 0:
            cls_msg = ""
            cls_hi_msg = ""
            if cfg.conflict_loss_weight > 0 and "conflict" in out:
                with torch.no_grad():
                    c_logits = out["conflict"][-1].squeeze(-1)
                    pred_unsat = c_logits > 0
                    target_unsat = gt_conflict_pre
                    tp = int((pred_unsat & target_unsat).sum().item())
                    fp = int((pred_unsat & ~target_unsat).sum().item())
                    fn = int((~pred_unsat & target_unsat).sum().item())
                    tn = int((~pred_unsat & ~target_unsat).sum().item())
                    P = tp / max(tp + fp, 1)
                    R = tp / max(tp + fn, 1)
                    # High-fill subset (fill > 0.9): the regime that matters
                    # most for tree-search use.
                    fill = (state.sum(dim=-1) == 1).float().mean(dim=-1)  # [B]
                    hi = fill > 0.9
                    n_hi = int(hi.sum().item())
                    if n_hi > 0:
                        tp_hi = int((pred_unsat & target_unsat & hi).sum().item())
                        fp_hi = int((pred_unsat & ~target_unsat & hi).sum().item())
                        fn_hi = int((~pred_unsat & target_unsat & hi).sum().item())
                        tn_hi = int((~pred_unsat & ~target_unsat & hi).sum().item())
                        P_hi = tp_hi / max(tp_hi + fp_hi, 1)
                        R_hi = tp_hi / max(tp_hi + fn_hi, 1)
                        cls_hi_msg = (f"cls@fill>0.9:P={P_hi:.2f}/R={R_hi:.2f}"
                                      f"[tp={tp_hi}/fp={fp_hi}/tn={tn_hi}/fn={fn_hi}]"
                                      f"  n_hi={n_hi}")
                    else:
                        cls_hi_msg = "cls@fill>0.9:n_hi=0"
                cls_msg = f"cls:P={P:.2f}/R={R:.2f}[tp={tp}/fp={fp}/tn={tn}/fn={fn}]"
            with torch.no_grad():
                # Cell accuracy: at multi-alive cells, BCE-argmax-over-alive vs GT.
                # Use the AUG-frame state and orig_y so the comparison is in
                # the same frame as `out["bce"]` (which was produced by the
                # grad forward on aug-frame state).
                aug_state_logged = aug_info["aug_state"]
                aug_orig_y_logged = aug_info["aug_orig_y"]
                multi_alive = (aug_state_logged.sum(dim=-1) > 1.5)
                masked = out["bce"][-1].masked_fill(
                    ~(aug_state_logged > 0.5), float("-inf"),
                )
                pred = masked.argmax(dim=-1)
                gt_argmax = aug_orig_y_logged.argmax(dim=-1)
                n_unc = int(multi_alive.sum().item())
                n_cor = int(((pred == gt_argmax) & multi_alive).sum().item())
                cell_acc = n_cor / max(n_unc, 1)
            min_d = int(pool_age.min().item())
            med_d = int(pool_age.median().item())
            avg_d = float(pool_age.float().mean().item())
            max_d = int(pool_age.max().item())
            pool_alpha = _alpha_surviving(pool_state, pool_solutions, pool_last_alpha)
            sat_frac = float((~_gt_conflict(pool_state, pool_alpha)).float().mean().item())
            # Three-line block: header (step/loss/lr), step counters, cls/pool.
            print(
                f"step={step:5d}/{cfg.steps}  loss={loss.item():.4f}  "
                f"acc={cell_acc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )
            print(
                f"    deduce={info['n_deduced']}  decide={info['n_decided']}  "
                f"solved={info['n_solved']}  "
                f"conflict={info['n_conflict']}(empty={info['n_conflict_empty']},"
                f"cls={info['n_conflict_cls']})  "
                f"age(min/med/avg/max)={min_d}/{med_d}/{avg_d:.1f}/{max_d}  "
                f"pool_sat={sat_frac:.2f}",
                flush=True,
            )
            if pool_use_snap:
                # E2 pool health under backtrack-matched training: decision-
                # depth distribution across the pool + cumulative restores.
                dep = pool_depth.float()
                d_p50 = int(pool_depth.median().item())
                d_p90 = int(torch.quantile(dep, 0.9).item())
                d_max = int(pool_depth.max().item())
                age_p50 = int(pool_age.median().item())
                age_p90 = int(torch.quantile(pool_age.float(), 0.9).item())
                print(
                    f"    [e2-pool bt={cfg.backtrack}] "
                    f"depth(p50/p90/max)={d_p50}/{d_p90}/{d_max}  "
                    f"age(p50/p90)={age_p50}/{age_p90}  "
                    f"restores_total={n_pool_restore_total}",
                    flush=True,
                )
            if cls_msg or cls_hi_msg:
                print(
                    f"    {cls_msg}  {cls_hi_msg}",
                    flush=True,
                )

        # Record the step-1 boundary so we can split compile vs post-compile
        # wallclock. Done at end-of-step-1 after any in-step eval/log have
        # also run (step 1 has neither in default config).
        if step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_step1_end = time.perf_counter()
            step1_compile_secs = t_step1_end - t_loop_start

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loop_end = time.perf_counter()
    train_total_secs = t_loop_end - t_loop_start
    train_post_compile_secs = (t_loop_end - t_step1_end) - intrain_eval_secs
    print(
        f"\nTrain wallclock: total={train_total_secs:.1f}s  "
        f"step1_compile={step1_compile_secs:.1f}s  "
        f"intrain_eval={intrain_eval_secs:.1f}s  "
        f"post_compile_train={train_post_compile_secs:.1f}s",
        flush=True,
    )

    extra = {
        "n_params": n_params, "n_examples_seen": n_examples_seen,
        "n_solved_total": n_solved_total, "n_tp_conflict_total": n_tp_conflict_total,
        "train_total_secs": train_total_secs,
        "train_step1_compile_secs": step1_compile_secs,
        "train_intrain_eval_secs": intrain_eval_secs,
        "train_post_compile_secs": train_post_compile_secs,
    }
    if ema is not None:
        extra["ema_state_dict"] = ema.state_dict()
        extra["ema_decay"] = cfg.ema_decay
    save_checkpoint(
        path=ckpt_path, model=model, model_cfg=cfg.model, train_cfg=cfg,
        extra=extra,
    )
    print(f"Saved: {ckpt_path}", flush=True)
    dataset.close()
    return ckpt_path
