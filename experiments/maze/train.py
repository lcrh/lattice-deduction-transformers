"""Pool trainer for Maze: the shared loop plus K-path targets and resumable snapshots."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from ldt.models.looped_transformer import PowersetModel
from ldt.training.utils.checkpoint import atomic_save, save_checkpoint
from ldt.training.utils.clock import TrainClock
from ldt.training.utils.scheduler import make_cosine_scheduler

from experiments.maze.data import MazeDataset, grid_dims, rescore
from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.train import (
    TrainConfig as PoolTrainConfig, _alpha_surviving, _gt_conflict, _given_mask, _losses,
    cell_accuracy, conflict_log, format_monitor, log_step, monitor_solve,
)

if TYPE_CHECKING:
    from experiments.maze.config import MazeExperiment


def maze_step(**kw) -> StepConfig:
    """Maze's operator: dihedral augmentation only (the five channels are not interchangeable)."""
    return StepConfig(augment_dihedral=True, permute_digits=False, **kw)


@dataclass
class TrainConfig(PoolTrainConfig):
    steps: int = 20000
    batch_size: int = 192
    step: StepConfig = field(default_factory=lambda: maze_step(threshold=0.1, augment=True))
    # Pool size over batch_size; a larger pool keeps older, harder entries around longer.
    pool_size_mult: float = 2.0
    checkpoint_every: int = 1000        # steps between `<ckpt>.resume.pt` snapshots; 0 = none
    keep_step_snapshots: bool = False   # also keep a `.step<N>.pt` copy of each snapshot


def train(exp: MazeExperiment, ckpt_path: str | Path,
          commit: Callable[[], None] = lambda: None) -> Path:
    """Train `exp` and save the checkpoint to `ckpt_path`, resuming from a snapshot if one exists.

    `commit` persists snapshots (a volume commit on Modal).
    """
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
        eval_h, eval_w = grid_dims(eval_data_cfg)
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

    start_step = 1
    prior_wallclock = None
    if resume_path is not None and resume_path.exists():
        print(f"[resume] loading snapshot from {resume_path}", flush=True)
        # CPU first: RNG states must stay CPU ByteTensors; the optimizer moves its
        # state to the parameters' device.
        rckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"])
        optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        scheduler.load_state_dict(rckpt["scheduler_state_dict"])
        if ema is not None:
            ema.shadow = {k: v.to(device) for k, v in rckpt["ema_state_dict"].items()}
        start_step = int(rckpt["step"]) + 1
        prior_wallclock = rckpt["train_wallclock"]
        torch.set_rng_state(rckpt["torch_rng"])
        if rckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state(rckpt["cuda_rng"])
        print(f"[resume] resumed at step {start_step} "
              f"(of {cfg.steps}; {cfg.steps - start_step + 1} remaining)", flush=True)
    elif resume_path is not None:
        print(f"[resume] no snapshot at {resume_path}; starting from step 1",
              flush=True)

    # Leftover puzzles from the last dataset batch, so a small backfill does
    # not discard most of an expensive K-path batch.
    buf_x = buf_sols = None

    def fresh_batch(n: int):
        """(x, solutions [n, K, S, C]) for n fresh puzzles."""
        nonlocal buf_x, buf_sols
        while buf_x is None or buf_x.shape[0] < n:
            x_b, sols_b, _ = dataset.next_batch()
            if buf_x is None:
                buf_x, buf_sols = x_b, sols_b
            else:
                buf_x = torch.cat([buf_x, x_b], dim=0)
                buf_sols = torch.cat([buf_sols, sols_b], dim=0)
        x = buf_x[:n].to(device).float()
        solutions = buf_sols[:n].to(device).float()
        buf_x, buf_sols = buf_x[n:], buf_sols[n:]
        return x, solutions

    pool_size = max(cfg.batch_size, int(round(cfg.batch_size * cfg.pool_size_mult)))
    pool_orig_x, pool_solutions = fresh_batch(pool_size)
    pool_state = pool_orig_x.clone()
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P, S, C]
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  (bs={cfg.batch_size}, mult={cfg.pool_size_mult})  "
          f"augment={cfg.step.augment}", flush=True)

    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0

    clock = TrainClock(prior_wallclock)
    for step in range(start_step, cfg.steps + 1):
        clock.tick()
        sample_idx = torch.randperm(pool_size, device=device)[:cfg.batch_size]
        state = pool_state[sample_idx]
        solutions = pool_solutions[sample_idx]   # [B, K, S, C]
        last_alpha = pool_last_alpha[sample_idx]  # [B, S, C]
        orig_x = pool_orig_x[sample_idx]
        age = pool_age[sample_idx]
        given_mask = _given_mask(orig_x)
        orig_y = _alpha_surviving(state, solutions, last_alpha)
        gt_conflict_pre = _gt_conflict(state, orig_y)
        is_sat_pre = ~gt_conflict_pre

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(model, state, given_mask, cfg.step, orig_y=orig_y)
        loss = _losses(out, aug_info["aug_state"], aug_info["aug_orig_y"],
                       aug_info["aug_given_mask"], is_sat_pre, gt_conflict_pre, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n_examples_seen += cfg.batch_size

        model.eval()
        want_info_stats = (step % cfg.log_every == 0)
        with torch.no_grad():
            new_state, detected_conflict, solved, _, info = dpll_step(
                model, state, given_mask, cfg.step,
                want_stats=want_info_stats,
            )
        gt_conflict_post = _gt_conflict(new_state, orig_y)

        new_age = age + 1
        age_exceeded = new_age > cfg.max_age
        true_positive_conflict = detected_conflict & gt_conflict_post
        discard = solved | true_positive_conflict | age_exceeded

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())

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

        pool_state[sample_idx] = new_state
        pool_orig_x[sample_idx] = new_orig_x
        pool_solutions[sample_idx] = new_solutions
        pool_last_alpha[sample_idx] = new_last_alpha
        pool_age[sample_idx] = new_age

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
                "train_wallclock": clock.finish(),
                **exp.checkpoint_meta(),        # so step snapshots can be evaluated directly
            }
            atomic_save(ckpt_payload, resume_path)
            if cfg.keep_step_snapshots:
                atomic_save(ckpt_payload, ckpt_path.with_suffix(f".step{step:07d}.pt"))
            commit()
            print(f"  [resume ckpt] saved at step {step} -> {resume_path}", flush=True)

        if cfg.monitor_every > 0 and step % cfg.monitor_every == 0:
            with clock.monitoring():
                res = monitor_solve(model, ema, eval_x, eval_y, eval_given, cfg)
                rescore(res, eval_y, eval_h, eval_w)     # any valid shortest path counts
                print(format_monitor(res, step, cfg.monitor_max_rounds), flush=True)

        if step % cfg.log_every == 0:
            with torch.no_grad():
                cls_msg = (conflict_log(out, gt_conflict_pre, state)
                           if cfg.conflict_loss_weight > 0 and "conflict" in out else "")
                cell_acc = cell_accuracy(out["bce"][-1], aug_info["aug_state"],
                                         aug_info["aug_orig_y"])
            pool_alpha = _alpha_surviving(pool_state, pool_solutions, pool_last_alpha)
            sat_frac = float((~_gt_conflict(pool_state, pool_alpha)).float().mean().item())
            log_step(step, cfg, loss, cell_acc, scheduler.get_last_lr()[0], info, pool_age,
                     sat_frac, cls_msg)
            # This step's outcomes bucketed by entry age, with the mean filled fraction.
            with torch.no_grad():
                new_fill = (new_state.sum(dim=-1) == 1).float().mean(dim=-1)  # [B]
            age_labels = ["a=0", "a=1", "a=2-3", "a=4-7", "a=8-15", "a=16-31", "a=32+"]
            age_bins = torch.bucketize(age, torch.tensor([1, 2, 4, 8, 16, 32], device=device))
            parts: list[str] = []
            for b in range(len(age_labels)):
                mask = (age_bins == b)
                n_b = int(mask.sum().item())
                if n_b == 0:
                    continue
                n_s = int((solved & mask).sum().item())
                n_d = int((detected_conflict & mask).sum().item())
                n_g = int((gt_conflict_post & mask).sum().item())
                f_b = float(new_fill[mask].mean().item())
                parts.append(f"{age_labels[b]}={n_b}@f{f_b:.2f}({n_s}s/{n_d}c/{n_g}g)")
            if parts:
                print(f"    age: {' '.join(parts)}", flush=True)

    extra = {
        "train_wallclock": clock.finish(),
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
