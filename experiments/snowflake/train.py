"""Pool trainer for Snowflake: the shared loop on a covering grid.

The state carries a locked in-puzzle channel (and optional constraint-group
channels) after the 6 vocab channels; losses and targets ignore out-of-puzzle
cells.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn.functional as F

from lattice_diffusion.models.looped_transformer import PowersetModel
from lattice_diffusion.models.weighted_bce import weighted_bce_with_logits
from lattice_diffusion.training.utils.checkpoint import save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.train import (
    TrainConfig as PoolTrainConfig, _alpha_surviving, _gt_conflict, _given_mask,
    cell_accuracy, conflict_log, format_monitor, log_step, monitor_solve,
)

from experiments.snowflake.data import (
    MAX_CONSTRAINT_GROUPS, VOCAB, SnowflakeDataset, _load_puzzles, load_states, split_name,
    split_path,
)

if TYPE_CHECKING:
    from experiments.snowflake.config import SnowflakeExperiment


N_CHANNELS = VOCAB + 1      # vocab + the in-puzzle channel


def n_channels(constraint_group_embed: bool) -> int:
    """State width, including constraint-group channels when enabled."""
    return N_CHANNELS + (MAX_CONSTRAINT_GROUPS if constraint_group_embed else 0)


def snowflake_step(**kw) -> StepConfig:
    """Snowflake's operator: digit permutation of the 6 vocab channels only (the grid is hex)."""
    return StepConfig(vocab_dim=VOCAB, augment_dihedral=False, **kw)


@dataclass
class TrainConfig(PoolTrainConfig):
    steps: int = 1000
    step: StepConfig = field(default_factory=lambda: snowflake_step(augment=True))


def _build_state(orig_x: torch.Tensor, mask: torch.Tensor,
                 groups: torch.Tensor | None = None) -> torch.Tensor:
    """[B, S, C] state: 6 vocab channels, the in-puzzle channel, then optional group channels."""
    pieces = [orig_x, mask.unsqueeze(-1).float()]
    if groups is not None:
        pieces.append(groups.float())
    return torch.cat(pieces, dim=-1)


def _losses(out, state, orig_y, given_mask, is_sat, gt_conflict_target, in_puzzle_mask,
            cfg: TrainConfig):
    """Sudoku's losses on the vocab channels, with out-of-puzzle cells masked out.

    The BCE is a mean over all B*S*VOCAB bits, so its scale matches Sudoku's.
    """
    B, S, C = state.shape
    device = state.device
    state_v = state[..., :VOCAB]
    orig_y_v = orig_y[..., :VOCAB]
    bce_target = state_v * orig_y_v
    pos_w = torch.full((B, 1, 1), cfg.bce_pos_mult, device=device)
    neg_w = torch.full((B, 1, 1), cfg.bce_neg_mult, device=device)
    softmax_w, conflict_w = cfg.softmax_loss_weight, cfg.conflict_loss_weight
    ip_chan = in_puzzle_mask.unsqueeze(-1).float()             # [B, S, 1]
    multi_alive_target = orig_y_v.sum(dim=-1) > 1              # [B, S]

    n_loops = len(out["bce"])
    iterations = range(n_loops) if cfg.supervise == "all" else (n_loops - 1,)
    total = torch.zeros((), device=device)
    for i in iterations:
        bce_logits = out["bce"][i][..., :VOCAB]                # [B, S, VOCAB]
        per_bit = weighted_bce_with_logits(
            bce_logits, bce_target, pos_w, neg_w, reduction="none",
        )
        total = total + (per_bit * ip_chan).mean()

        sm_logits = out["softmax"][i][..., :VOCAB]              # [B, S, VOCAB]
        gt_idx = orig_y_v.argmax(dim=-1)                        # [B, S]
        ce_mask = ~given_mask & ~multi_alive_target & in_puzzle_mask & is_sat.unsqueeze(-1)
        if ce_mask.any():
            total = total + softmax_w * F.cross_entropy(
                sm_logits[ce_mask], gt_idx[ce_mask],
            )

        if "conflict" in out and conflict_w > 0:
            c_logits = out["conflict"][i].squeeze(-1)
            total = total + conflict_w * F.binary_cross_entropy_with_logits(
                c_logits, gt_conflict_target.float(),
            )
    return total / len(iterations)


def _monitor_set(path: str, n: int, device, group_embed: bool):
    """A seeded sample of `n` validation puzzles as (state, y, given_mask, in_puzzle_mask)."""
    puzzles = _load_puzzles(path)
    rng = np.random.default_rng(200)
    idx = rng.permutation(len(puzzles))[:n]
    x, y, mask, groups = load_states(puzzles, idx, rng if group_embed else None)
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    groups = groups.to(device) if groups is not None else None
    return _build_state(x, mask, groups), y, _given_mask(x), mask


def train(exp: SnowflakeExperiment, ckpt_path: str | Path,
          commit: Callable[[], None] = lambda: None) -> Path:
    """Train `exp` and save the checkpoint to `ckpt_path`.

    `commit` is unused: runs are short enough to restart instead of resuming.
    """
    cfg = exp.train
    ckpt_path = Path(ckpt_path)
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

    dataset = SnowflakeDataset(dataclasses.replace(exp.data, batch_size=cfg.batch_size))

    if cfg.monitor_every > 0:
        monitor_split = split_name(exp.data.dataset, "val")
        eval_state, eval_y, eval_given, eval_in_puzzle = _monitor_set(
            str(split_path(exp.data.data_dir, exp.data.dataset, "val")), cfg.monitor_n_puzzles,
            device, exp.data.constraint_group_embed)
        print(f"Monitoring set: {eval_state.shape[0]} puzzles from {monitor_split} "
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

    def fresh_batch(n: int):
        """(x, solutions [n, 1, S, VOCAB], in-puzzle mask, full state) for n fresh puzzles."""
        bx, by, bm, bg = [], [], [], []
        while sum(t.shape[0] for t in bx) < n:
            x_b, y_b, m_b, _, g_b = dataset.next_batch()
            for buf, t in zip((bx, by, bm, bg), (x_b, y_b, m_b, g_b)):
                buf.append(t)
        x = torch.cat(bx, dim=0)[:n].to(device).float()
        y = torch.cat(by, dim=0)[:n].to(device).float()
        m = torch.cat(bm, dim=0)[:n].to(device).bool()
        g = torch.cat(bg, dim=0)[:n].to(device) if bg[0] is not None else None
        solutions = y.unsqueeze(1)  # [n, 1, S, VOCAB]
        return x, solutions, m, _build_state(x, m, g)

    pool_size = cfg.batch_size
    pool_orig_x, pool_solutions, pool_in_puzzle, pool_state = fresh_batch(pool_size)
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P, S, VOCAB]
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  (= bs={cfg.batch_size})  "
          f"augment={cfg.step.augment} (digit-perm only; "
          f"vocab_dim={cfg.step.vocab_dim})", flush=True)


    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0

    for step in range(1, cfg.steps + 1):
        sample_idx = torch.randperm(pool_size, device=device)[:cfg.batch_size]
        state = pool_state[sample_idx]
        solutions = pool_solutions[sample_idx]      # [B, K, S, VOCAB]
        last_alpha = pool_last_alpha[sample_idx]    # [B, S, VOCAB]
        orig_x = pool_orig_x[sample_idx]
        in_puzzle_mask = pool_in_puzzle[sample_idx]
        age = pool_age[sample_idx]
        given_mask = _given_mask(orig_x)
        orig_y = _alpha_surviving(state[..., :VOCAB], solutions, last_alpha, in_puzzle_mask)
        gt_conflict_pre = _gt_conflict(state[..., :VOCAB], orig_y, in_puzzle_mask)
        is_sat_pre = ~gt_conflict_pre

        # Pad orig_y to the state width so the frame can permute it like the state.
        orig_y_padded = F.pad(orig_y, (0, state.shape[-1] - VOCAB))   # [B, S, C]

        # Digit permutation only, so the in-puzzle mask is the same in the augmented frame.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(model, state, given_mask, cfg.step, orig_y=orig_y_padded)
        loss = _losses(out, aug_info["aug_state"], aug_info["aug_orig_y"],
                       aug_info["aug_given_mask"], is_sat_pre, gt_conflict_pre,
                       in_puzzle_mask, cfg)
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
                in_puzzle_mask=in_puzzle_mask,
                want_stats=want_info_stats,
            )
        gt_conflict_post = _gt_conflict(new_state[..., :VOCAB], orig_y, in_puzzle_mask)

        new_age = age + 1
        age_exceeded = new_age > cfg.max_age
        true_positive_conflict = detected_conflict & gt_conflict_post
        discard = solved | true_positive_conflict | age_exceeded

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())

        n_to_replace = int(discard.sum().item())
        new_last_alpha = orig_y.clone()
        if n_to_replace > 0:
            fx, f_solutions, fm, f_state = fresh_batch(n_to_replace)
            new_state = new_state.clone()
            new_orig_x = orig_x.clone()
            new_solutions = solutions.clone()
            new_in_puzzle = in_puzzle_mask.clone()
            new_age = new_age.clone()
            new_state[discard] = f_state
            new_orig_x[discard] = fx
            new_solutions[discard] = f_solutions
            new_in_puzzle[discard] = fm
            new_age[discard] = 0
            new_last_alpha[discard] = f_solutions.bool().any(dim=1).float()
        else:
            new_orig_x = orig_x
            new_solutions = solutions
            new_in_puzzle = in_puzzle_mask

        pool_state[sample_idx] = new_state
        pool_orig_x[sample_idx] = new_orig_x
        pool_solutions[sample_idx] = new_solutions
        pool_in_puzzle[sample_idx] = new_in_puzzle
        pool_last_alpha[sample_idx] = new_last_alpha
        pool_age[sample_idx] = new_age

        if cfg.monitor_every > 0 and step % cfg.monitor_every == 0:
            res = monitor_solve(model, ema, eval_state, eval_y, eval_given, cfg,
                                in_puzzle_mask=eval_in_puzzle)
            print(format_monitor(res, step, cfg.monitor_max_rounds), flush=True)

        if step % cfg.log_every == 0:
            with torch.no_grad():
                cls_msg = (conflict_log(out, gt_conflict_pre, None)
                           if cfg.conflict_loss_weight > 0 and "conflict" in out else "")
                cell_acc = cell_accuracy(out["bce"][-1][..., :VOCAB],
                                         aug_info["aug_state"][..., :VOCAB],
                                         aug_info["aug_orig_y"], in_puzzle_mask)
            pool_v = pool_state[..., :VOCAB]
            pool_alpha = _alpha_surviving(pool_v, pool_solutions, pool_last_alpha, pool_in_puzzle)
            sat_frac = float((~_gt_conflict(pool_v, pool_alpha, pool_in_puzzle))
                             .float().mean().item())
            log_step(step, cfg, loss, cell_acc, scheduler.get_last_lr()[0], info, pool_age,
                     sat_frac, cls_msg)

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
