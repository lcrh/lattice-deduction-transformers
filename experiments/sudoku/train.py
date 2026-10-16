"""Pool-based trainer, shared by all benchmarks.

Each step trains on a batch of pool states, advances them with `dpll_step`,
and replaces solved, truly conflicted or aged-out entries with fresh puzzles.
Snowflake and Maze reuse the config base, targets, losses and logging here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.nn.functional as F

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeDataset, load_puzzles
from lattice_diffusion.eval import indices as index_lists
from lattice_diffusion.models.looped_transformer import PowersetModel
from lattice_diffusion.models.weighted_bce import weighted_bce_with_logits
from lattice_diffusion.training.utils.checkpoint import save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.solve import SolveConfig, solve

if TYPE_CHECKING:
    from experiments.sudoku.config import SudokuExperiment


@dataclass
class TrainConfig:
    """Pool-trainer settings; Snowflake and Maze subclass it to change defaults."""

    steps: int = 4000
    batch_size: int = 512
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
    # Deep supervision: average the loss over every loop's readout, or only the final one.
    supervise: Literal["all", "final"] = "all"

    step: StepConfig = field(default_factory=lambda: StepConfig(augment=True))
    max_age: int = 100      # training steps an entry stays in the pool
    compile: bool = True

    # EMA of the weights, saved with the checkpoint and swapped in at eval.
    use_ema: bool = False
    ema_decay: float = 0.999

    log_every: int = 20
    # Tight-budget solve on validation puzzles every `monitor_every` steps.
    # Logging only: it runs under a forked RNG, so it never changes the weights.
    monitor_every: int = 100
    monitor_n_puzzles: int = 200
    monitor_max_rounds: int = 5
    monitor_n_chains: int = 64


def _gt_conflict(
    state: torch.Tensor,            # [B, S, C]
    orig_y: torch.Tensor,           # [B, S, C] (one-hot for K=1; multi-alive α for K>1)
    in_puzzle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B] True where some in-puzzle cell has no candidate left that `orig_y` allows."""
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
    """[B, S, C] union of the solutions still consistent with `state`.

    When none survives, falls back to `last_alpha` (else `solutions[:, 0]`), so a
    step that kills the last surviving solution still gets a conflict target.
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


def monitor_solve(model, ema, x, y, given_mask, cfg, **solve_kw):
    """Tight-budget solve with the training operator, under a forked RNG and EMA weights if any."""
    model.eval()
    ema_backup = ema.swap_in(model) if ema is not None else None
    devices = [x.device] if x.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        res = solve(model, x, y, given_mask, SolveConfig(
            step=cfg.step, max_rounds=cfg.monitor_max_rounds, n_chains=cfg.monitor_n_chains,
            # Match training batch shape so torch.compile reuses its kernel.
            batch_size=cfg.batch_size,
        ), **solve_kw)
    if ema_backup is not None:
        ema.swap_out(model, ema_backup)
    return res


def format_monitor(res, step: int, max_rounds: int) -> str:
    """One log line summarizing a monitoring solve."""
    unsound = res.diag_total_unsound_deductions / max(res.diag_total_deduced, 1)
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)
    return (f"  [monitor step={step}, max_rounds={max_rounds}] "
            f"correct={int(res.correct.sum())}/{res.solved.shape[0]}  "
            f"wrong={int(res.wrong.sum())}  to={int(res.timeouts.sum())}  "
            f"calls={res.model_calls}  unsound={unsound:.3%}  "
            f"cls:P={cls_p:.2f}/R={cls_r:.2f}")


def _given_mask(orig_x: torch.Tensor) -> torch.Tensor:
    """[B, S] bool: cells that started as singletons in the original puzzle."""
    return (orig_x.sum(dim=-1) == 1)


def _losses(out, state, orig_y, given_mask, is_sat, gt_conflict_target, cfg: TrainConfig):
    """Weighted BCE + softmax CE + conflict BCE, averaged over the supervised loops."""
    B, S, C = state.shape
    device = state.device
    bce_target = state * orig_y
    pos_w = torch.full((B, 1, 1), cfg.bce_pos_mult, device=device)
    neg_w = torch.full((B, 1, 1), cfg.bce_neg_mult, device=device)
    softmax_w, conflict_w = cfg.softmax_loss_weight, cfg.conflict_loss_weight

    # No softmax target where several solutions survive at a cell.
    multi_alive_target = orig_y.sum(dim=-1) > 1                          # [B, S]

    n_loops = len(out["bce"])
    iterations = range(n_loops) if cfg.supervise == "all" else (n_loops - 1,)
    total = torch.zeros((), device=device)
    for i in iterations:
        total = total + weighted_bce_with_logits(out["bce"][i], bce_target, pos_w, neg_w)

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
    return total / len(iterations)


def conflict_pr(pred: torch.Tensor, target: torch.Tensor,
                subset: torch.Tensor | None = None) -> str:
    """Conflict-head precision / recall against true conflicts, with the raw counts."""
    if subset is not None:
        pred, target = pred[subset], target[subset]
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    tn = int((~pred & ~target).sum().item())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return f"P={precision:.2f}/R={recall:.2f}[tp={tp}/fp={fp}/tn={tn}/fn={fn}]"


def conflict_log(out: dict, gt_conflict: torch.Tensor, state: torch.Tensor | None) -> str:
    """Conflict head vs. true conflicts, overall and (given `state`) on >90%-filled puzzles."""
    pred = out["conflict"][-1].squeeze(-1) > 0
    msg = f"cls:{conflict_pr(pred, gt_conflict)}"
    if state is None:
        return msg
    hi = (state.sum(dim=-1) == 1).float().mean(dim=-1) > 0.9
    n_hi = int(hi.sum().item())
    if n_hi == 0:
        return f"{msg}  cls@fill>0.9:n_hi=0"
    return f"{msg}  cls@fill>0.9:{conflict_pr(pred, gt_conflict, hi)}  n_hi={n_hi}"


def cell_accuracy(bce_logits: torch.Tensor, state: torch.Tensor, orig_y: torch.Tensor,
                  mask: torch.Tensor | None = None) -> float:
    """At multi-candidate cells, how often the top BCE logit among alive candidates is right."""
    multi_alive = state.sum(dim=-1) > 1.5
    if mask is not None:
        multi_alive = multi_alive & mask
    pred = bce_logits.masked_fill(~(state > 0.5), float("-inf")).argmax(dim=-1)
    n_cor = int(((pred == orig_y.argmax(dim=-1)) & multi_alive).sum().item())
    return n_cor / max(int(multi_alive.sum().item()), 1)


def log_step(step: int, cfg: TrainConfig, loss: torch.Tensor, cell_acc: float, lr: float,
             info: dict, pool_age: torch.Tensor, sat_frac: float, cls_msg: str = "") -> None:
    """The periodic three-line training log."""
    ages = (f"{int(pool_age.min().item())}/{int(pool_age.median().item())}/"
            f"{float(pool_age.float().mean().item()):.1f}/{int(pool_age.max().item())}")
    print(f"step={step:5d}/{cfg.steps}  loss={loss.item():.4f}  acc={cell_acc:.4f}  "
          f"lr={lr:.6f}", flush=True)
    print(f"    deduce={info['n_deduced']}  decide={info['n_decided']}  "
          f"solved={info['n_solved']}  "
          f"conflict={info['n_conflict']}(empty={info['n_conflict_empty']},"
          f"cls={info['n_conflict_cls']})  "
          f"age(min/med/avg/max)={ages}  pool_sat={sat_frac:.2f}", flush=True)
    if cls_msg:
        print(f"    {cls_msg}", flush=True)


def _monitor_set(cache_dir: str, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """First `n` puzzles of the held-out `sudoku_val` list."""
    idx = index_lists.load("sudoku_val").indices[:n]
    x, y = load_puzzles(cache_dir, "train", idx)
    return x.to(device).float(), y.to(device).float()


def train(exp: SudokuExperiment, ckpt_path: str | Path,
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

    dataset = SudokuExtremeDataset(dataclasses.replace(exp.data, batch_size=cfg.batch_size))

    if cfg.monitor_every > 0:
        eval_x, eval_y = _monitor_set(exp.data.cache_dir, cfg.monitor_n_puzzles, device)
        eval_given = (eval_x.sum(dim=-1) == 1)
        print(f"Monitoring set: {eval_x.shape[0]} sudoku_val puzzles "
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
        """(x, solutions) for n fresh puzzles; solutions has a K=1 axis."""
        bx, by = [], []
        while sum(t.shape[0] for t in bx) < n:
            x_b, y_b, _ = dataset.next_batch()
            bx.append(x_b); by.append(y_b)
        x = torch.cat(bx, dim=0)[:n].to(device).float()
        y = torch.cat(by, dim=0)[:n].to(device).float()
        solutions = y.unsqueeze(1)  # [n, K=1, S, C]
        return x, solutions

    # Pool tensors are canonical-frame. `orig_x` is the entry's puzzle,
    # `state` its current lattice, `last_alpha` the previous step's target.
    pool_size = cfg.batch_size
    pool_orig_x, pool_solutions = fresh_batch(pool_size)
    pool_state = pool_orig_x.clone()
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P, S, C]
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    # Latent carried between an entry's steps (carry h / z); zeroed on backfill.
    pool_carry = (torch.zeros(pool_size, pool_state.shape[1], exp.model.dim, device=device)
                  if exp.model.carries_state else None)
    print(f"Pool size: {pool_size}  (= bs={cfg.batch_size})  "
          f"augment={cfg.step.augment}", flush=True)

    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0

    for step in range(1, cfg.steps + 1):
        sample_idx = torch.randperm(pool_size, device=device)[:cfg.batch_size]
        state = pool_state[sample_idx]
        solutions = pool_solutions[sample_idx]   # [B, K, S, C]
        last_alpha = pool_last_alpha[sample_idx]  # [B, S, C]
        orig_x = pool_orig_x[sample_idx]
        age = pool_age[sample_idx]
        carry_kw = None
        if exp.model.carry != "off":
            carry_kw = {"orig_x": orig_x if exp.model.needs_puzzle else None,
                        "carry": pool_carry[sample_idx] if pool_carry is not None else None}
        given_mask = _given_mask(orig_x)
        orig_y = _alpha_surviving(state, solutions, last_alpha)
        gt_conflict_pre = _gt_conflict(state, orig_y)
        is_sat_pre = ~gt_conflict_pre

        # Loss in a fresh augmented frame, against targets mapped into it.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(model, state, given_mask, cfg.step,
                                    orig_y=orig_y, carry_kw=carry_kw)
        loss = _losses(out, aug_info["aug_state"], aug_info["aug_orig_y"],
                       aug_info["aug_given_mask"], is_sat_pre, gt_conflict_pre, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n_examples_seen += cfg.batch_size

        # Advance the batch with the solver's operator (its own fresh frame).
        model.eval()
        want_info_stats = (step % cfg.log_every == 0)
        with torch.no_grad():
            new_state, detected_conflict, solved, _, info = dpll_step(
                model, state, given_mask, cfg.step,
                want_stats=want_info_stats, carry_kw=carry_kw,
            )
        new_carry = info["carry_out"].clone() if pool_carry is not None else None
        gt_conflict_post = _gt_conflict(new_state, orig_y)

        # Discard solved, truly conflicted and aged-out entries.
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
            if new_carry is not None:
                new_carry[discard] = 0
        else:
            new_orig_x = orig_x
            new_solutions = solutions

        pool_state[sample_idx] = new_state
        pool_orig_x[sample_idx] = new_orig_x
        pool_solutions[sample_idx] = new_solutions
        pool_last_alpha[sample_idx] = new_last_alpha
        pool_age[sample_idx] = new_age
        if pool_carry is not None:
            pool_carry[sample_idx] = new_carry

        if cfg.monitor_every > 0 and step % cfg.monitor_every == 0:
            res = monitor_solve(model, ema, eval_x, eval_y, eval_given, cfg)
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
