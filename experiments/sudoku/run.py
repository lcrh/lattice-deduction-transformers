"""Modal entry point for sudoku: train then evaluate.

Usage:
    uv run modal run experiments/sudoku/run.py
    uv run modal run experiments/sudoku/run.py --steps 2000 --batch-size 512
"""

import dataclasses
import json
import time
from dataclasses import asdict

import modal
import torch
from torch import nn


# Keep in sync with run() signature defaults. Used for the run-config table
# at the top of every run, with non-defaults highlighted.
_RUN_PARAMS: tuple[tuple[str, object], ...] = (
    ("steps",                    4000),
    ("batch_size",               512),
    ("n_eval_puzzles",           200),
    ("n_train_puzzles",          None),
    ("seed",                     0),
    ("bce_pos_mult",             4.0),
    ("bce_neg_mult",             0.5),
    ("softmax_loss_weight",      0.2),
    ("conflict_loss_weight",     0.1),
    ("weight_decay",             0.1),
    ("lr",                       3e-3),
    ("max_age",                  100),
    ("warmup_fraction",          0.1),
    ("threshold",                0.10),
    ("temp_decide",              1.5),
    ("cls_threshold",            0.5),
    ("eval_cls_threshold",       0.6),
    ("eval_max_rounds",          1000),
    ("eval_n_chains",            64),
    ("eval_batch_size",          512),
    ("augment",                  True),
    ("data_augment_digit_perm",  True),
    ("data_augment_dihedral",    True),
    ("use_ema",                  False),
    ("ema_decay",                0.999),
    ("estimate_sequential",      False),
    ("seq_drain_max_rounds",     200),
    ("eval_dropout_p",           0.05),
    ("n_loops",                  16),
    ("pre_norm",                 True),
)


def _print_run_config(values: dict) -> None:
    """Tabular dump of run() args at startup. Non-default values are starred."""
    BOLD, RESET = "\033[1m", "\033[0m"
    print("=" * 64, flush=True)
    print(f"RUN CONFIG  (* = non-default; values bolded if your terminal supports ANSI)",
          flush=True)
    print("=" * 64, flush=True)
    n_changed = 0
    for name, default in _RUN_PARAMS:
        val = values.get(name, "<unset>")
        is_changed = (val != default)
        if is_changed:
            n_changed += 1
            marker = "*"
            shown = f"{BOLD}{val!r}{RESET}"
            tail = f"   (default {default!r})"
        else:
            marker = " "
            shown = f"{val!r}"
            tail = ""
        print(f"  {marker} {name:<28} = {shown}{tail}", flush=True)
    print(f"  {n_changed}/{len(_RUN_PARAMS)} non-default", flush=True)
    print("=" * 64, flush=True)

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig, SudokuExtremeDataset
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve
from experiments.sudoku.train import TrainConfig, train


app = modal.App("sudoku")


@app.function(
    image=image,
    gpu="B200",
    timeout=3600 * 4,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    steps: int = 4000,
    batch_size: int = 512,
    n_eval_puzzles: int = 200,
    n_train_puzzles: int | None = None,
    seed: int = 0,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 0.1,
    weight_decay: float = 0.1,
    lr: float = 3e-3,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    eval_cls_threshold: float = 0.6,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = 512,
    augment: bool = True,
    data_augment_digit_perm: bool = True,
    data_augment_dihedral: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_loops: int = 16,
    pre_norm: bool = True,
):
    # Snapshot the call-site arg values BEFORE any local mutation, then dump
    # the config table. (Snapshot locals() outside the comprehension —
    # Python 3 comprehensions have their own scope, so locals() inside one
    # only sees the iter-vars, NOT the function's parameters.)
    _loc_snapshot = dict(locals())
    _arg_values = {name: _loc_snapshot[name] for name, _ in _RUN_PARAMS}
    _print_run_config(_arg_values)

    ts = time.strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
    )
    model_cfg = LoopedTransformerConfig(
        cls_token=conflict_loss_weight > 0,
        n_loops=n_loops,
        pre_norm=pre_norm,
    )
    ckpt_path = train(TrainConfig(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        bce_pos_mult=bce_pos_mult,
        bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        warmup_fraction=warmup_fraction,
        step=step_cfg,
        max_age=max_age,
        use_ema=use_ema,
        ema_decay=ema_decay,
        model=model_cfg,
        data=SudokuExtremeConfig(
            cache_dir=DATA_MOUNT, batch_size=batch_size, seed=42,
            n_puzzles=n_train_puzzles,
            augment_digit_perm=data_augment_digit_perm,
            augment_dihedral=data_augment_dihedral,
        ),
        out_dir=f"{CHECKPOINT_MOUNT}/sudoku",
        name=f"seed{seed}_{steps}s_bs{batch_size}_aug{int(augment)}_{ts}",
    ))
    checkpoint_volume.commit()

    print("\n" + "=" * 60, flush=True)
    print(f"Eval ({n_eval_puzzles} test puzzles)", flush=True)
    print("=" * 60, flush=True)

    ckpt = load_checkpoint(str(ckpt_path))
    cfg_loaded = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg_loaded)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    # If the checkpoint includes EMA weights, swap them into the model
    # in place (eval-only — we don't restore live weights afterward).
    # Note: save_checkpoint does `data.update(extra)`, so `ema_state_dict`
    # lives at the top level of the loaded dict, not nested under "extra".
    swap_in_ema_if_present(model, ckpt)
    if eval_dropout_p > 0.0:
        n_drop = 0
        n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = eval_dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = eval_dropout_p
                n_mha += 1
        model.train()
        print(f"  Dropout-noise eval: overrode {n_drop} nn.Dropout + "
              f"{n_mha} MHA-internal-attn-dropout layers to p={eval_dropout_p}, "
              f"model in train() mode", flush=True)

    # Eval data: seed=200, zero_hint_weight=1.0 (all SAT), n=n_eval_puzzles.
    eval_ds = SudokuExtremeDataset(SudokuExtremeConfig(
        cache_dir=DATA_MOUNT, split="test", n_puzzles=n_eval_puzzles,
        batch_size=n_eval_puzzles, seed=200,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        augment_digit_perm=False, augment_dihedral=False,
    ))
    x, y, sat = eval_ds.next_batch(); eval_ds.close()
    sat_mask = sat.bool()
    x = x[sat_mask].to(device).float()
    y = y[sat_mask].to(device).float()
    given_mask = (x.sum(dim=-1) == 1)
    n_sat = x.shape[0]
    print(f"  Loaded {n_sat}/{n_eval_puzzles} SAT eval puzzles", flush=True)

    # Build a separate eval-time step_cfg that may use a different
    # cls_threshold than training. Training uses `cls_threshold` (default
    # 0.5); final eval uses `eval_cls_threshold` (default 0.6), tuned on
    # the train set.
    eval_step_cfg = dataclasses.replace(step_cfg, cls_threshold=eval_cls_threshold)
    solve_cfg = SolveConfig(
        step=eval_step_cfg, max_rounds=eval_max_rounds,
        n_chains=eval_n_chains, batch_size=eval_batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
    )
    res = solve(model, x, y, given_mask, solve_cfg)

    n = res.solved.shape[0]
    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    avg_rounds_solved = float(
        res.round_solved[res.solved].float().mean().item()
        if int(res.solved.sum().item()) > 0 else 0.0
    )
    avg_resets = float(res.n_resets.float().mean().item())

    # Diagnostics
    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    print(f"\n{'='*60}\nRESULT SUMMARY\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}  "
          f"n_chains={res.n_chains}", flush=True)
    print(f"  Total model calls: {res.model_calls}  "
          f"(amortized: {res.model_calls / max(n_correct, 1):.1f} calls/correct)",
          flush=True)
    print(f"  Avg rounds-to-solve (winning chain): {avg_rounds_solved:.1f}  "
          f"Avg resets/puzzle: {avg_resets:.2f}", flush=True)
    print(f"  Deduction soundness: {res.diag_total_unsound_deductions} unsound / "
          f"{res.diag_total_deduced} deduced  (rate={unsound_rate:.4%})", flush=True)
    print(f"  Conflict head (vs gt-conflict-post-deduce): "
          f"P={cls_p:.3f} R={cls_r:.3f} "
          f"[tp={res.diag_conflict_tp} fp={res.diag_conflict_fp} "
          f"fn={res.diag_conflict_fn} tn={res.diag_conflict_tn}] "
          f"over {res.diag_active_chain_rounds} active chain-rounds",
          flush=True)
    print(f"{'='*60}", flush=True)

    train_wallclock = {
        "total_secs": ckpt.get("train_total_secs"),
        "step1_compile_secs": ckpt.get("train_step1_compile_secs"),
        "intrain_eval_secs": ckpt.get("train_intrain_eval_secs"),
        "post_compile_secs": ckpt.get("train_post_compile_secs"),
    }

    eval_json_path = ckpt_path.with_suffix(".eval.json")
    eval_json_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "n_eval_puzzles": n,
        "n_chains": res.n_chains,
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "model_calls_total": res.model_calls,
        "avg_rounds_solved": avg_rounds_solved,
        "avg_resets": avg_resets,
        "step_cfg": asdict(step_cfg),
        "max_rounds": eval_max_rounds,
        "train_wallclock": train_wallclock,
        "diag": {
            "total_deduced": res.diag_total_deduced,
            "total_unsound_deductions": res.diag_total_unsound_deductions,
            "unsound_rate": unsound_rate,
            "conflict_tp": res.diag_conflict_tp,
            "conflict_fp": res.diag_conflict_fp,
            "conflict_fn": res.diag_conflict_fn,
            "conflict_tn": res.diag_conflict_tn,
            "conflict_precision": cls_p,
            "conflict_recall": cls_r,
            "active_chain_rounds": res.diag_active_chain_rounds,
        },
    }, indent=2))

    # Per-puzzle JSONL dump for downstream analysis. First line is a metadata
    # header (with the same summary as eval.json plus full run config), then
    # one line per puzzle with its outcome.
    eval_jsonl_path = ckpt_path.with_suffix(".eval.jsonl")
    n_givens_per_puzzle = given_mask.sum(dim=-1).long().tolist()
    with eval_jsonl_path.open("w") as fh:
        fh.write(json.dumps({
            "kind": "header",
            "checkpoint": str(ckpt_path),
            "n_eval_puzzles": n,
            "n_chains": res.n_chains,
            "max_rounds": eval_max_rounds,
            "step_cfg": asdict(step_cfg),
            "run_args": {name: _arg_values[name] for name, _ in _RUN_PARAMS},
            "train_wallclock": train_wallclock,
            "summary": {
                "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
                "model_calls_total": res.model_calls,
                "avg_rounds_solved": avg_rounds_solved,
                "avg_resets": avg_resets,
                "unsound_rate": unsound_rate,
                "conflict_p": cls_p, "conflict_r": cls_r,
            },
        }) + "\n")
        for i in range(n):
            is_correct = bool(res.correct[i].item())
            is_wrong = bool(res.wrong[i].item())
            is_timeout = bool(res.timeouts[i].item())
            rs = int(res.round_solved[i].item())
            # forwards_unbatched: per-puzzle cost in single-chain forwards
            # if we ran with no batching of any kind (M=1 slot, K=1 chain,
            # serial). Solved: K*(round_solved+1). Wrong/timeout: K*max_rounds.
            if is_correct:
                forwards_unbatched = (rs + 1) * eval_n_chains
            else:
                forwards_unbatched = eval_max_rounds * eval_n_chains
            fh.write(json.dumps({
                "kind": "puzzle",
                "puzzle_idx": i,
                "correct": is_correct,
                "wrong": is_wrong,
                "timeout": is_timeout,
                "round_solved": rs,
                "n_resets": int(res.n_resets[i].item()),
                "n_givens": int(n_givens_per_puzzle[i]),
                "puzzle_calls": int(res.puzzle_calls[i].item()),
                "forwards_unbatched": forwards_unbatched,
            }) + "\n")
    checkpoint_volume.commit()

    return {
        "steps": steps, "batch_size": batch_size,
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "n_chains": res.n_chains,
        "checkpoint": str(ckpt_path),
    }


@app.local_entrypoint()
def entrypoint(
    steps: int = 4000,
    batch_size: int = 512,
    n_eval_puzzles: int = 200,
    n_train_puzzles: int | None = None,
    seed: int = 0,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 0.1,
    weight_decay: float = 0.1,
    lr: float = 3e-3,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    eval_cls_threshold: float = 0.6,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = 512,
    augment: bool = True,
    data_augment_digit_perm: bool = True,
    data_augment_dihedral: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_loops: int = 16,
    pre_norm: bool = True,
):
    result = run.remote(
        steps=steps, batch_size=batch_size,
        n_eval_puzzles=n_eval_puzzles, n_train_puzzles=n_train_puzzles, seed=seed,
        bce_pos_mult=bce_pos_mult, bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        weight_decay=weight_decay,
        lr=lr,
        max_age=max_age,
        warmup_fraction=warmup_fraction,
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        eval_cls_threshold=eval_cls_threshold,
        eval_max_rounds=eval_max_rounds,
        eval_n_chains=eval_n_chains,
        eval_batch_size=eval_batch_size,
        augment=augment,
        data_augment_digit_perm=data_augment_digit_perm,
        data_augment_dihedral=data_augment_dihedral,
        use_ema=use_ema,
        ema_decay=ema_decay,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        eval_dropout_p=eval_dropout_p,
        n_loops=n_loops,
        pre_norm=pre_norm,
    )
    print(f"\nFinal: {result}", flush=True)
