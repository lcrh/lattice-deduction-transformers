"""Pool-based trainer for maze.

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

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from experiments.maze.data import MazeDataset, grid_dims, rescore
from lattice_diffusion.models.looped_transformer import PowersetModel
from lattice_diffusion.training.utils.checkpoint import save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.train import (
    _alpha_surviving, _gt_conflict, _given_mask, _losses, format_monitor, monitor_solve,
)

if TYPE_CHECKING:
    from experiments.maze.config import MazeExperiment


@dataclass
class TrainConfig:
    steps: int = 20000
    batch_size: int = 192
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1   # warmup_steps = int(steps * warmup_fraction)
    seed: int = 0

    bce_pos_mult: float = 4.0
    bce_neg_mult: float = 0.5
    softmax_loss_weight: float = 0.2
    conflict_loss_weight: float = 0.1

    # Maze's operator: dihedral augmentation only (the 5 channels — wall /
    # free / start / goal / path — are not interchangeable, so no digit-perm).
    step: StepConfig = field(default_factory=lambda: maze_step(threshold=0.5, augment=True))
    max_age: int = 100  # age = train steps since backfill (counts up regardless of sampling)
    # Pool-size multiplier over batch_size. >1.0 gives a larger persistent
    # pool than the per-step sample, so older / harder entries hang around
    # longer (variety + curriculum effect).
    pool_size_mult: float = 2.0
    compile: bool = True

    # EMA of trainable params, evaluated at eval time. TRM paper recommends
    # decay=0.999. Shadow + live state are both saved — eval auto-swaps EMA
    # into the live model if `ema_state_dict` is present in the checkpoint.
    use_ema: bool = False
    ema_decay: float = 0.999

    log_every: int = 20
    # In-training monitoring: tight-budget mini-solve on held-out puzzles
    # (Maze-Hard test split, or a seed-200 synthetic pool); logging only,
    # see sudoku/train.py.
    monitor_every: int = 100
    monitor_n_puzzles: int = 200
    monitor_max_rounds: int = 5
    monitor_n_chains: int = 64

    # Every `checkpoint_every` steps a full {model, optimizer, scheduler,
    # step, RNG} snapshot is written next to the checkpoint
    # (`<ckpt>.resume.pt`, plus a `<ckpt>.step<N>.pt` copy); a restarted run
    # resumes from it. 0 disables snapshots.
    checkpoint_every: int = 1000


def maze_step(**kw) -> StepConfig:
    return StepConfig(augment_dihedral=True, permute_digits=False, **kw)


def train(exp: MazeExperiment, ckpt_path: str | Path,
          commit: Callable[[], None] = lambda: None) -> Path:
    """Train `exp` and save the checkpoint to `ckpt_path`, resuming from a
    snapshot if a previous attempt left one; `commit` persists snapshots."""
    cfg = exp.train
    ckpt_path = Path(ckpt_path)
    resume_path = ckpt_path.with_suffix(".resume.pt") if cfg.checkpoint_every > 0 else None
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PowersetModel(exp.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  steps={cfg.steps}  bs={cfg.batch_size}",
          flush=True)
    if cfg.compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)

    dataset = MazeDataset(dataclasses.replace(exp.data, batch_size=cfg.batch_size))

    if cfg.monitor_every > 0:
        # Maze-Hard: the test split. Synthetic: a separate seed-200 pool.
        eval_data_cfg = dataclasses.replace(
            exp.data, split="test" if exp.data.dataset == "maze_hard" else "train",
            n_puzzles=cfg.monitor_n_puzzles, batch_size=cfg.monitor_n_puzzles, seed=200,
            k_solutions=1, augment_dihedral=False, augment_swap_endpoints=False,
            cache_suffix="",
        )
        _H_eval, _W_eval = grid_dims(eval_data_cfg)
        eval_ds = MazeDataset(eval_data_cfg)
        ex_x, ex_sols, ex_sat = eval_ds.next_batch()
        eval_ds.close()
        sat_mask = ex_sat.bool()
        eval_x = ex_x[sat_mask][:cfg.monitor_n_puzzles].to(device).float()
        eval_y = ex_sols[sat_mask, 0][:cfg.monitor_n_puzzles].to(device).float()
        eval_given = (eval_x.sum(dim=-1) == 1)
        print(f"Monitoring set: {eval_x.shape[0]} puzzles "
              f"(every {cfg.monitor_every} steps, "
              f"K={cfg.monitor_n_chains}, max_rounds={cfg.monitor_max_rounds})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay, betas=cfg.betas)
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_fraction))
    scheduler = make_cosine_scheduler(optimizer, cfg.steps, warmup_steps)

    ema: ModelEMA | None = None
    if cfg.use_ema:
        ema = ModelEMA(model, cfg.ema_decay)
        print(f"EMA enabled (decay={cfg.ema_decay}); shadowing "
              f"{len(ema.shadow)} param tensors", flush=True)

    # Resume support.
    start_step = 1
    if resume_path is not None and resume_path.exists():
        print(f"[resume] loading snapshot from {resume_path}", flush=True)
        rckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"])
        optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        scheduler.load_state_dict(rckpt["scheduler_state_dict"])
        if ema is not None and "ema_state_dict" in rckpt:
            ema.shadow = {k: v.to(device) for k, v in rckpt["ema_state_dict"].items()}
        start_step = int(rckpt["step"]) + 1
        # RNG restore is best-effort — a torch upgrade between save and load
        # can leave the saved state in an incompatible format
        # (`torch.set_rng_state` requires a uint8 ByteTensor; older saves
        # are sometimes a different dtype/shape). Failing the resume over
        # the RNG state would force throwing away thousands of training
        # steps, so we skip it on mismatch and continue with the current
        # RNG.
        try:
            torch.set_rng_state(rckpt["torch_rng"])
            if torch.cuda.is_available() and rckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(rckpt["cuda_rng"])
        except (TypeError, RuntimeError) as e:
            print(f"[resume] RNG state restore skipped: {e}", flush=True)
        print(f"[resume] resumed at step {start_step} "
              f"(of {cfg.steps}; {cfg.steps - start_step + 1} remaining)", flush=True)
    elif resume_path is not None:
        print(f"[resume] no snapshot at {resume_path}; starting from step 1",
              flush=True)

    # Leftover-puzzle buffer so we don't waste a whole dataset batch when
    # only a few entries are backfilled per step. The K-paths sampler does
    # ~512 puzzles' worth of work per batch — without this buffer, each
    # ~50-puzzle backfill call would force the sampler to produce a fresh
    # 512-batch, throttling step rate even with the threaded prefetch.
    ds_buf: list[torch.Tensor | None] = [None, None]   # [x, solutions]
    def fresh_batch(n: int):
        """Return `(x, solutions)` with `solutions: [n, K, S, C]`.

        K-paths sampling lives in the dataset's prefetch thread (see
        `MazeDataset._k_prefetch_loop`); this just consumes from the
        leftover buffer and the queue.
        """
        while ds_buf[0] is None or ds_buf[0].shape[0] < n:
            x_b, sols_b, _ = dataset.next_batch()
            if ds_buf[0] is None:
                ds_buf[0], ds_buf[1] = x_b, sols_b
            else:
                ds_buf[0] = torch.cat([ds_buf[0], x_b], dim=0)
                ds_buf[1] = torch.cat([ds_buf[1], sols_b], dim=0)
        x = ds_buf[0][:n].to(device).float()
        solutions = ds_buf[1][:n].to(device).float()
        ds_buf[0] = ds_buf[0][n:]
        ds_buf[1] = ds_buf[1][n:]
        return x, solutions

    # Pool: orig_x fixed per entry; state evolves; `solutions: [P, K, S, C]`
    # carries the K ground truths. Each step computes α(surviving) on the
    # fly via `_alpha_surviving` to feed into BCE / gt_conflict, falling back
    # to `last_alpha` (the previous step's α) when |surviving|=0.
    pool_size = max(cfg.batch_size, int(round(cfg.batch_size * cfg.pool_size_mult)))
    pool_orig_x, pool_solutions = fresh_batch(pool_size)
    pool_state = pool_orig_x.clone()
    # Fresh-state α: all K solutions survive (orig_x is permissive at every
    # cell), so OR-of-surviving = OR over all K. For K=1 = solutions[:, 0].
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P, S, C]
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  (bs={cfg.batch_size}, mult={cfg.pool_size_mult})  "
          f"augment={cfg.step.augment}", flush=True)

    # Let `aug_forward` build cell_perms lazily on its first call (it caches
    # nothing, but the cost is one perm-table build per device which is
    # negligible). dpll_step has its own _CELL_PERMS_CACHE for the no-grad
    # path. Pass None and let aug_forward derive (H, W) from `state.shape[1]`.
    cell_perms_train = None

    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0

    for step in range(start_step, cfg.steps + 1):
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
        # `orig_y` for this step = α(surviving K solutions), falling back
        # to `last_alpha` (the previous step's α) when |surviving|=0. At
        # K=1 with one-hot solutions this is just the canonical GT;
        # multi-alive at branching cells when K>1.
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
        # `permute_digits=False` mirrors dpll_step: digit-perm becomes
        # identity (vocab_dim=0). Maze channels (wall/free/S/G/path) aren't
        # interchangeable so we only do dihedral here.
        aug_vocab_dim = 0 if not cfg.step.permute_digits else cfg.step.vocab_dim
        out, aug_info = aug_forward(
            model, state, given_mask, orig_y=orig_y,
            augment=cfg.step.augment,
            augment_dihedral=cfg.step.augment_dihedral,
            cell_perms=cell_perms_train,
            vocab_dim=aug_vocab_dim,
            return_all=True,
        )
        loss = _losses(
            out,
            aug_info["aug_state"], aug_info["aug_orig_y"], aug_info["aug_given_mask"],
            is_sat_pre, gt_conflict_pre,
            cfg.bce_pos_mult, cfg.bce_neg_mult,
            cfg.softmax_loss_weight, cfg.conflict_loss_weight,
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
        discard = solved | true_positive_conflict | age_exceeded

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())


        # ---- backfill discarded sampled entries ----
        # Pool tensors are CANONICAL; new_state is canonical (dpll_step
        # inverted any aug); fresh data is canonical (the dataset has aug
        # off in our config). Straight canonical assignment.
        # `new_last_alpha` defaults to this step's `orig_y` (correct fallback
        # for next step); backfilled entries get OR-over-K of fresh solutions.
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

        # ---- periodic resume snapshot + unique step checkpoint ----
        if resume_path is not None and step % cfg.checkpoint_every == 0:
            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": (ema.shadow if ema is not None else None),
                "step": step,
                "model_cfg": asdict(exp.model),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state()
                              if torch.cuda.is_available() else None),
            }
            torch.save(ckpt_payload, resume_path)
            # Also keep a per-step copy so intermediate steps can be probed
            # later (the resume snapshot is overwritten each time).
            unique_path = ckpt_path.with_suffix(f".step{step:07d}.pt")
            torch.save(ckpt_payload, unique_path)
            commit()
            print(f"  [resume ckpt] saved at step {step} → {resume_path}  "
                  f"(also: {unique_path})", flush=True)

        # ---- in-train monitoring (any valid minimal path counts) ----
        if cfg.monitor_every > 0 and step % cfg.monitor_every == 0:
            res = monitor_solve(model, ema, eval_x, eval_y, eval_given, cfg)
            rescore(res, eval_y, _H_eval, _W_eval)
            print(format_monitor(res, step, cfg.monitor_max_rounds), flush=True)

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
            if cls_msg or cls_hi_msg:
                print(
                    f"    {cls_msg}  {cls_hi_msg}",
                    flush=True,
                )
            # Pool-depth-bucketed step metrics. `age` is the depth at the
            # start of this step (= number of prior dpll_steps applied to
            # the entry; 0 = just backfilled). Solved/conflict/gt-conflict
            # counts come from THIS step's dpll_step, bucketed by depth.
            # Also report avg fill (fraction of cells that are singletons in
            # the post-step state) per bucket — shows how filled-in puzzles
            # at each depth typically are.
            with torch.no_grad():
                # Use post-step state (new_state in canonical frame) for fill.
                new_fill = (new_state.sum(dim=-1) == 1).float().mean(dim=-1)  # [B]
            depth_boundaries = torch.tensor(
                [1, 2, 4, 8, 16, 32], device=device,
            )
            depth_labels = ["d=0", "d=1", "d=2-3", "d=4-7", "d=8-15", "d=16-31", "d=32+"]
            depth_bins = torch.bucketize(age, depth_boundaries, right=False)
            parts: list[str] = []
            for b in range(len(depth_labels)):
                mask = (depth_bins == b)
                n_b = int(mask.sum().item())
                if n_b == 0:
                    continue
                n_s = int((solved & mask).sum().item())
                n_d = int((detected_conflict & mask).sum().item())
                n_g = int((gt_conflict_post & mask).sum().item())
                f_b = float(new_fill[mask].mean().item())
                parts.append(f"{depth_labels[b]}={n_b}@f{f_b:.2f}({n_s}s/{n_d}c/{n_g}g)")
            if parts:
                print(f"    depth: {' '.join(parts)}", flush=True)

    extra = {
        "n_params": n_params, "n_examples_seen": n_examples_seen,
        "n_solved_total": n_solved_total, "n_tp_conflict_total": n_tp_conflict_total,
    }
    if ema is not None:
        extra["ema_state_dict"] = ema.state_dict()
        extra["ema_decay"] = cfg.ema_decay
    extra.update(exp.checkpoint_meta())
    save_checkpoint(
        path=ckpt_path, model=model, model_cfg=exp.model, train_cfg=cfg,
        extra=extra,
    )
    print(f"Saved: {ckpt_path}", flush=True)
    dataset.close()
    return ckpt_path
