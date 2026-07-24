"""Pool-based trainer for snowflake.

Mirrors `experiments/sudoku/train.py` but adapts for snowflake's
covering-grid setup:
  - State carries the in-puzzle mask as a 7th channel
    (`state[..., :6]` = vocab powerset, `state[..., 6]` = in-puzzle bit
    locked at 1 for in-puzzle cells, 0 elsewhere). The model has
    `n_channels=7` so the mask is just an extra "always-on" vocab slot
    the operator never touches.
  - All deduce / decide / loss / GT-conflict logic gates on
    `in_puzzle_mask` so out-of-puzzle cells are inert.
  - Augmentation is digit-perm only (no spatial dihedral): the covering
    grid is hex (S=150 not a perfect square) and dihedral on the
    covering would not preserve hex topology. `cfg.step.augment_dihedral`
    is set False; `cfg.step.vocab_dim=6` so digit-perm only shuffles
    the first 6 channels.

Pool / discard policy is identical to sudoku. Same `dpll_step` is
used (imported from `experiments.sudoku.dpll`) — its mask-aware
extensions (`in_puzzle_mask`, `vocab_dim`, `augment_dihedral`) are
default-preserving for sudoku.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.models.weighted_bce import weighted_bce_with_logits
from lattice_diffusion.training.utils.checkpoint import save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.solve import SolveConfig, solve

from experiments.snowflake.data import SnowflakeConfig, SnowflakeDataset


# Snowflake recipe constants (must stay consistent with data.py).
VOCAB = 6           # vocab channels (digits 1..6)
N_CHANNELS = 7      # vocab + 1 mask channel
GRID_ROWS = 15
GRID_COLS = 10
SEQ_LEN = GRID_ROWS * GRID_COLS  # 150


def _default_step_cfg() -> StepConfig:
    """StepConfig with snowflake-specific defaults: vocab_dim=6, augment_dihedral=False."""
    return StepConfig(vocab_dim=VOCAB, augment_dihedral=False)


def _default_model_cfg() -> LoopedTransformerConfig:
    """LoopedTransformerConfig with snowflake-specific defaults: n_channels=7,
    grid_rows=15, grid_cols=10, cls_token=True (for the conflict head)."""
    return LoopedTransformerConfig(
        n_channels=N_CHANNELS,
        seq_len=SEQ_LEN,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        cls_token=True,
    )


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1
    seed: int = 0

    bce_pos_mult: float = 4.0
    bce_neg_mult: float = 0.5
    softmax_loss_weight: float = 0.2
    conflict_loss_weight: float = 1.0

    step: StepConfig = field(default_factory=_default_step_cfg)
    max_age: int = 100
    compile: bool = True

    use_ema: bool = False
    ema_decay: float = 0.999

    log_every: int = 20
    eval_every: int = 100
    eval_n_puzzles: int = 200
    eval_max_rounds: int = 5
    eval_n_chains: int = 64

    out_dir: str = "checkpoints/snowflake"
    name: str = ""
    # When True, the checkpoint path is deterministic (`{name}.pt`, no
    # wallclock timestamp) and `train()` refuses to overwrite an existing
    # file unless `overwrite=True`. Default False preserves the legacy
    # timestamped `{name}_{ts}.pt` behavior.
    no_timestamp: bool = False
    overwrite: bool = False

    model: LoopedTransformerConfig = field(default_factory=_default_model_cfg)
    data: SnowflakeConfig = field(default_factory=SnowflakeConfig)
    # Optional separate eval dataset config. If None, we snapshot from cfg.data
    # with seed=200 + zero_hint_weight=1.0 (eval is all SAT puzzles).
    eval_data: SnowflakeConfig | None = None


def _build_state(orig_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build [B, S, 7] state from snowflake [B, S, 6] vocab + [B, S] mask.

    Channel 6 is the in-puzzle mask, locked at 1 for in-puzzle cells.
    Out-of-puzzle cells have all-zero state (vocab + mask).
    """
    return torch.cat([orig_x, mask.unsqueeze(-1).float()], dim=-1)


def _gt_conflict(
    state: torch.Tensor,            # [B, S, C] (vocab + mask)
    orig_y: torch.Tensor,           # [B, S, vocab_dim] (one-hot for K=1, multi-alive α for K>1)
    in_puzzle_mask: torch.Tensor,   # [B, S]
) -> torch.Tensor:
    """[B] bool: True iff any in-puzzle cell has no state bit consistent with orig_y.

    For one-hot orig_y (K=1): equivalent to "any GT bit dead at an
    in-puzzle cell". For multi-alive α(surviving) orig_y (K>1): "any
    in-puzzle cell where state ∩ α has no alive bit".
    """
    state_v = state[..., :VOCAB]                                    # vocab-only view
    consistent = (state_v * orig_y).any(dim=-1)                     # [B, S]
    return ~(consistent | ~in_puzzle_mask).all(dim=-1)


def _alpha_surviving(
    state: torch.Tensor,        # [B, S, C]
    solutions: torch.Tensor,    # [B, K, S, vocab_dim]
    in_puzzle_mask: torch.Tensor,
    last_alpha: torch.Tensor | None = None,  # [B, S, vocab_dim]
) -> torch.Tensor:
    """α(surviving K solutions). Returns [B, S, vocab_dim] float.

    Snowflake variant: state has a locked mask channel beyond vocab; we
    compare against `state[..., :VOCAB]` only. When |surviving|=0, falls
    back to `last_alpha` (the α from the previous step, when |surviving|
    was non-empty) — see sudoku/train.py:_alpha_surviving for the full
    derivation. For K=1 with one-hot solutions, this returns
    `solutions[:, 0]`, the canonical static GT.
    """
    sol_bool = solutions > 0.5                                          # [B, K, S, V]
    state_v = state[..., :VOCAB] > 0.5                                  # [B, S, V]
    consistent = (state_v.unsqueeze(1) & sol_bool).any(dim=-1)          # [B, K, S]
    consistent = consistent | ~in_puzzle_mask.unsqueeze(1)
    surviving = consistent.all(dim=-1)                                   # [B, K]
    any_surviving = surviving.any(dim=-1)                                # [B]
    alpha = (sol_bool & surviving.unsqueeze(-1).unsqueeze(-1)).any(dim=1)  # [B, S, V]
    fallback = last_alpha if last_alpha is not None else sol_bool[:, 0].float()
    alpha = torch.where(
        any_surviving.unsqueeze(-1).unsqueeze(-1),
        alpha.float(),
        fallback,
    )
    return alpha


def _given_mask(orig_x: torch.Tensor) -> torch.Tensor:
    """[B, S] bool: cells given as singletons in the original puzzle.

    `orig_x` is [B, S, vocab_dim] (vocab channels only, no mask). Out-of-
    puzzle cells have all-zero `orig_x` so they fail `sum==1` naturally.
    """
    return (orig_x.sum(dim=-1) == 1)


def _losses(
    out, state, orig_y, given_mask, is_sat, gt_conflict_target,
    bce_pos_mult, bce_neg_mult, softmax_w, conflict_w,
    in_puzzle_mask,  # [B, S] — gates BCE/softmax to in-puzzle cells
):
    """Compute losses over vocab channels only, gated by in_puzzle_mask.

    `state` and `orig_y` arrive as [B, S, N_CHANNELS=7] (orig_y is padded
    with a zero-valued mask channel by the caller so that aug functions
    can permute it homogeneously with state). `out["bce"][i]` and
    `out["softmax"][i]` are also [B, S, N_CHANNELS]. We slice all three
    to `[..., :VOCAB]` for loss math so the auxiliary mask channel gets
    no gradient signal (the model learns to ignore it).

    BCE reduction matches sudoku's `weighted_bce_with_logits(...).mean()`
    semantics: per-bit loss multiplied by an in-puzzle mask, then mean
    over all elements so the loss scale is consistent with sudoku
    (divisor is `B*S*VOCAB`, numerator only counts in-puzzle bits).
    """
    B, S, C = state.shape
    device = state.device
    state_v = state[..., :VOCAB]                              # [B, S, VOCAB]
    orig_y_v = orig_y[..., :VOCAB]                            # [B, S, VOCAB]
    bce_target = state_v * orig_y_v                            # [B, S, VOCAB]
    pos_w = torch.full((B, 1, 1), bce_pos_mult, device=device)
    neg_w = torch.full((B, 1, 1), bce_neg_mult, device=device)

    # In-puzzle mask broadcastable to channels — zeros out per-bit BCE
    # contributions at out-of-puzzle cells before mean reduction.
    ip_chan = in_puzzle_mask.unsqueeze(-1).float()             # [B, S, 1]

    # Cells where α has multiple alive vocab channels — skip softmax CE
    # there (would arbitrarily pick argmax). For K=1 one-hot orig_y, this
    # mask is False everywhere → ce_mask covers all in-puzzle cells.
    multi_alive_target = orig_y_v.sum(dim=-1) > 1                       # [B, S]

    n_loops = len(out["bce"])
    total = torch.zeros((), device=device)
    for i in range(n_loops):
        bce_logits = out["bce"][i][..., :VOCAB]                # [B, S, VOCAB]
        per_bit = weighted_bce_with_logits(
            bce_logits, bce_target, pos_w, neg_w, reduction="none",
        )
        # `mean()` over [B, S, VOCAB] so divisor is constant across batches;
        # numerator gates by in_puzzle_mask. Sudoku's path is equivalent
        # (every cell is in-puzzle so `ip_chan` is all-ones and this collapses
        # to the original `weighted_bce_with_logits(...).mean()`).
        total = total + (per_bit * ip_chan).mean()

        sm_logits = out["softmax"][i][..., :VOCAB]              # [B, S, VOCAB]
        gt_idx = orig_y_v.argmax(dim=-1)                        # [B, S]
        # Softmax CE on non-given, single-alive-target, in-puzzle cells of
        # SAT puzzles only.
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
    return total / n_loops


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.no_timestamp:
        # Deterministic path used by launchers so downstream
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

    model = PowersetModel(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  steps={cfg.steps}  bs={cfg.batch_size}",
          flush=True)
    if cfg.compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)

    cfg.data.batch_size = cfg.batch_size
    dataset = SnowflakeDataset(cfg.data)

    # In-train eval set: take a snapshot of all-SAT puzzles with a fixed seed.
    # `cfg.eval_data` should point at a held-out test split. If it's None we
    # fall back to deriving the test parquet from cfg.data.data_path
    # (`..._train.parquet` -> `..._test.parquet`); failing that we sample from
    # the training file with a different seed and warn loudly. The first two
    # options yield clean held-out evaluation; the warning case will leak.
    if cfg.eval_data is not None:
        eval_cfg = cfg.eval_data
    else:
        train_path = cfg.data.data_path
        sibling = train_path.replace("_train.parquet", "_test.parquet").replace(
            "_train.json", "_test.json"
        )
        if sibling != train_path and Path(sibling).exists():
            print(f"  eval_data unset; using sibling test split at {sibling}",
                  flush=True)
            eval_cfg = SnowflakeConfig(
                data_path=sibling,
                n_puzzles=cfg.eval_n_puzzles, batch_size=cfg.eval_n_puzzles,
                seed=200,
                zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
            )
        else:
            print(f"  WARNING: eval_data unset AND no sibling test split found "
                  f"({sibling!r} does not exist); inline eval will sample from "
                  f"the training file with seed=200, expect leak.", flush=True)
            eval_cfg = SnowflakeConfig(
                data_path=train_path,
                n_puzzles=cfg.eval_n_puzzles, batch_size=cfg.eval_n_puzzles,
                seed=200,
                zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
            )
    eval_ds = SnowflakeDataset(eval_cfg)
    ex_x, ex_y, ex_mask, ex_sat = eval_ds.next_batch()
    eval_ds.close()
    sat_mask = ex_sat.bool()
    eval_x = ex_x[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    eval_y = ex_y[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    eval_in_puzzle = ex_mask[sat_mask][:cfg.eval_n_puzzles].to(device).bool()
    eval_state = _build_state(eval_x, eval_in_puzzle)
    eval_given = _given_mask(eval_x)
    print(f"In-train eval set: {eval_state.shape[0]} SAT puzzles "
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
        """Return `(x, solutions, mask)` with `solutions: [n, K=1, S, vocab_dim]`."""
        bx, by, bm = [], [], []
        while sum(t.shape[0] for t in bx) < n:
            x_b, y_b, m_b, _ = dataset.next_batch()
            bx.append(x_b); by.append(y_b); bm.append(m_b)
        x = torch.cat(bx, dim=0)[:n].to(device).float()
        y = torch.cat(by, dim=0)[:n].to(device).float()
        m = torch.cat(bm, dim=0)[:n].to(device).bool()
        solutions = y.unsqueeze(1)  # [n, 1, S, VOCAB]
        return x, solutions, m

    # Pool: stores canonical-frame state (with mask channel), orig_x (vocab
    # only), solutions ([P, K, S, VOCAB]), in_puzzle_mask, age.
    pool_size = cfg.batch_size
    pool_orig_x, pool_solutions, pool_in_puzzle = fresh_batch(pool_size)
    pool_state = _build_state(pool_orig_x, pool_in_puzzle)
    # Fresh-state α: all K solutions survive (orig_x permissive over vocab).
    # Equals OR-over-K-solutions per (cell, channel). For K=1 = solutions[:,0].
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P, S, VOCAB]
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  (= bs={cfg.batch_size})  "
          f"augment={cfg.step.augment} (digit-perm only; "
          f"vocab_dim={cfg.step.vocab_dim})", flush=True)

    # No dihedral cell perms — the covering grid is hex (S=150 isn't a
    # perfect square); aug_forward will skip the spatial step.
    cell_perms_train = None

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
        # `orig_y` for this step is dynamic α(surviving K solutions), with
        # last_alpha as fallback when |surviving|=0. For K=1 snowflake,
        # this equals `pool_solutions[sample_idx, 0]`.
        orig_y = _alpha_surviving(state, solutions, in_puzzle_mask, last_alpha)
        gt_conflict_pre = _gt_conflict(state, orig_y, in_puzzle_mask)
        is_sat_pre = ~gt_conflict_pre

        # Pad orig_y to N_CHANNELS so apply_aug_state's digit-perm gather
        # (sized to N_CHANNELS=7) works homogeneously on state and orig_y.
        # The trailing channel stays 0 (digit-perm has identity on slots
        # >=vocab_dim), and `_losses` slices back to vocab.
        orig_y_padded = F.pad(orig_y, (0, N_CHANNELS - VOCAB))   # [B, S, N_CHANNELS]

        # ---- forward + losses (grad-tracked; aug applied via aug_forward) ----
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(
            model, state, given_mask, orig_y=orig_y_padded,
            augment=cfg.step.augment,
            augment_dihedral=cfg.step.augment_dihedral,
            cell_perms=cell_perms_train,
            vocab_dim=cfg.step.vocab_dim,
            return_all=True,
        )
        # In-puzzle mask is permuted-by-cell only (no dihedral here), so it's
        # frame-invariant. aug_forward doesn't return aug_in_puzzle_mask;
        # since we only do digit-perm (no spatial), the mask is unchanged.
        aug_in_puzzle_mask = in_puzzle_mask
        # aug_orig_y comes back permuted in vocab dim — argmax indices change
        # frames but bce_target is a bitwise AND so still vocab-dim sliced.
        # is_sat is frame-invariant; gt_conflict_pre was computed pre-aug.
        loss = _losses(
            out,
            aug_info["aug_state"], aug_info["aug_orig_y"], aug_info["aug_given_mask"],
            is_sat_pre, gt_conflict_pre,
            cfg.bce_pos_mult, cfg.bce_neg_mult,
            cfg.softmax_loss_weight, cfg.conflict_loss_weight,
            aug_in_puzzle_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n_examples_seen += cfg.batch_size

        # ---- step state forward (no grad) ----
        model.eval()
        want_info_stats = (step % cfg.log_every == 0)
        with torch.no_grad():
            new_state, detected_conflict, solved, _, info = dpll_step(
                model, state, given_mask, cfg.step,
                orig_y=orig_y_padded,
                in_puzzle_mask=in_puzzle_mask,
                want_stats=want_info_stats,
            )
        gt_conflict_post = _gt_conflict(new_state, orig_y, in_puzzle_mask)

        # ---- discard policy on the sampled batch ----
        new_age = age + 1
        age_exceeded = new_age > cfg.max_age
        true_positive_conflict = detected_conflict & gt_conflict_post
        discard = solved | true_positive_conflict | age_exceeded

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())

        # ---- backfill discarded sampled entries ----
        n_to_replace = int(discard.sum().item())
        new_last_alpha = orig_y.clone()
        if n_to_replace > 0:
            fx, f_solutions, fm = fresh_batch(n_to_replace)
            new_state = new_state.clone()
            new_orig_x = orig_x.clone()
            new_solutions = solutions.clone()
            new_in_puzzle = in_puzzle_mask.clone()
            new_age = new_age.clone()
            new_state[discard] = _build_state(fx, fm)
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

        # ---- in-train eval ----
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            model.eval()
            ema_backup = ema.swap_in(model) if ema is not None else None
            with torch.no_grad():
                eval_solve_cfg = SolveConfig(
                    step=cfg.step,
                    max_rounds=cfg.eval_max_rounds,
                    n_chains=cfg.eval_n_chains,
                    batch_size=cfg.batch_size,
                )
                eval_res = solve(
                    model, eval_state, eval_y, eval_given, eval_solve_cfg,
                    in_puzzle_mask=eval_in_puzzle,
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

        # ---- logging ----
        if step % cfg.log_every == 0:
            cls_msg = ""
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
                cls_msg = f"cls:P={P:.2f}/R={R:.2f}[tp={tp}/fp={fp}/tn={tn}/fn={fn}]"
            with torch.no_grad():
                aug_state_logged = aug_info["aug_state"]
                aug_orig_y_logged = aug_info["aug_orig_y"]
                # Cell accuracy at multi-alive in-puzzle cells (vocab only).
                state_v = aug_state_logged[..., :VOCAB]
                multi_alive = (state_v.sum(dim=-1) > 1.5) & in_puzzle_mask
                masked = out["bce"][-1][..., :VOCAB].masked_fill(
                    ~(state_v > 0.5), float("-inf"),
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
            sat_frac = float(
                (~_gt_conflict(pool_state, _alpha_surviving(pool_state, pool_solutions, pool_in_puzzle, pool_last_alpha), pool_in_puzzle))
                .float().mean().item()
            )
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
            if cls_msg:
                print(f"    {cls_msg}", flush=True)

    extra = {
        "n_params": n_params, "n_examples_seen": n_examples_seen,
        "n_solved_total": n_solved_total, "n_tp_conflict_total": n_tp_conflict_total,
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
