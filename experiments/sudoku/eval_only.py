"""Eval an existing checkpoint with the hybrid-batched solver.

Eval data: seed=200, zero_hint_weight=1.0 (all SAT), n=200 (post-filter).

Usage:
    uv run modal run --detach experiments/sudoku/eval_only.py \
      --checkpoint /checkpoints/sudoku/seed0_4000s_bs512_aug1_<ts>.pt \
      --temp-eliminate 0.0
"""

import json
import time

import modal
import torch
from torch import nn

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


app = modal.App("sudoku-eval-only")


@app.function(
    image=image, gpu="B200", timeout=7200,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str,
    n_eval: int = 200,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    log_per_round_fill: bool = False,
    out_suffix: str = ".eval.fixed.json",
    split: str = "test",
    compile: bool = False,
    eval_max_timeouts: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print("RUN CONFIG  (eval_only)", flush=True)
    print(f"  checkpoint={checkpoint}", flush=True)
    print(f"  n_eval={n_eval} threshold={threshold} cls_threshold={cls_threshold} "
          f"eval_max_timeouts={eval_max_timeouts} out_suffix={out_suffix}",
          flush=True)
    print(f"Loading: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    cfg = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)
    if dropout_p > 0.0:
        n_drop = 0
        n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = dropout_p
                n_mha += 1
        model.train()
        print(f"  Dropout-noise eval: overrode {n_drop} nn.Dropout + "
              f"{n_mha} MHA-internal-attn-dropout layers to p={dropout_p}, "
              f"model in train() mode", flush=True)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    eval_ds = SudokuExtremeDataset(SudokuExtremeConfig(
        cache_dir=DATA_MOUNT, split=split, n_puzzles=n_eval,
        batch_size=n_eval, seed=200,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        augment_digit_perm=False, augment_dihedral=False,
    ))
    x_all, y_all, sat = eval_ds.next_batch(); eval_ds.close()
    sat_mask = sat.bool()
    x = x_all[sat_mask].to(device).float()
    y = y_all[sat_mask].to(device).float()
    given_mask = (x.sum(dim=-1) == 1)
    n = x.shape[0]
    print(f"Loaded {n}/{n_eval} SAT eval puzzles (matches HP run.py)", flush=True)

    # Eval uses deterministic threshold elimination (no stochastic temp).
    # `augment` toggle now lives on StepConfig — `dpll_step` handles
    # aug as a black box.
    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
    )
    _max_to = eval_max_timeouts if eval_max_timeouts > 0 else None
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=max_rounds,
        n_chains=n_chains, batch_size=batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        log_per_round_fill=log_per_round_fill,
        eval_max_timeouts=_max_to,
    )
    print(f"Solving {n} puzzles | n_chains={n_chains} batch_size={batch_size} "
          f"(M={batch_size//n_chains} puzzles/forward) max_rounds={max_rounds} "
          f"eval_max_timeouts={_max_to}",
          flush=True)
    print(f"  step: threshold={threshold} (deterministic) "
          f"temp_dec={temp_decide} cls_threshold={cls_threshold} "
          f"augment={augment}", flush=True)

    t0 = time.time()
    res = solve(model, x, y, given_mask, solve_cfg)
    elapsed = time.time() - t0

    # Clean gap-free prefix [0..k]: on abort, fast stragglers past a gap must
    # not inflate the denominator. puzzle_calls < 0 means never filled.
    outcomes: dict[int, dict] = {}
    for i in range(n):
        if int(res.puzzle_calls[i].item()) < 0:
            continue
        outcomes[i] = {
            "idx": i,
            "correct": bool(res.correct[i].item()),
            "wrong": bool(res.wrong[i].item()),
            "timeout": bool(res.timeouts[i].item()),
            "round_solved": int(res.round_solved[i].item()),
            "n_resets": int(res.n_resets[i].item()),
            "puzzle_calls": int(res.puzzle_calls[i].item()),
        }
    k = -1
    while (k + 1) in outcomes:
        k += 1
    prefix_idxs = range(0, k + 1)
    n_prefix = k + 1
    n_correct = sum(1 for i in prefix_idxs if outcomes[i]["correct"])
    n_wrong = sum(1 for i in prefix_idxs if outcomes[i]["wrong"])
    n_timeout = sum(1 for i in prefix_idxs if outcomes[i]["timeout"])
    avg_calls = res.model_calls / max(n_correct, 1)
    if res.aborted:
        print(f"  [prefix] gap-free prefix length n={n_prefix} "
              f"(aborted={res.aborted}, outcomes={len(outcomes)}/{n})",
              flush=True)

    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    print(f"\n{'='*60}\nRESULT (streaming-queue solver, {n_prefix} puzzles)\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n_prefix}  wrong={n_wrong}  timeouts={n_timeout}  "
          f"aborted={res.aborted}", flush=True)
    print(f"  total_calls={res.model_calls}  avg/correct={avg_calls:.1f}", flush=True)
    print(f"  Deduction soundness: {res.diag_total_unsound_deductions} unsound / "
          f"{res.diag_total_deduced} deduced  (rate={unsound_rate:.4%})", flush=True)
    print(f"  Conflict head P={cls_p:.3f} R={cls_r:.3f} "
          f"[tp={res.diag_conflict_tp} fp={res.diag_conflict_fp} "
          f"fn={res.diag_conflict_fn} tn={res.diag_conflict_tn}] "
          f"over {res.diag_active_chain_rounds} active chain-rounds", flush=True)
    print(f"  wall: {elapsed:.0f}s", flush=True)

    out = {
        "checkpoint": checkpoint,
        "n_eval": n_prefix,
        "n_eval_puzzles": n_prefix,
        "eval_aborted": res.aborted,
        "n_evaluated_prefix": n_prefix,
        "eval_max_timeouts": eval_max_timeouts,
        "solver": "hybrid_per_chunk",
        "solver_config": {
            "threshold": threshold, "temp_decide": temp_decide,
            "cls_threshold": cls_threshold,
            "n_chains": n_chains, "batch_size": batch_size,
            "max_rounds": max_rounds,
            "augment": augment,
            "eval_max_timeouts": eval_max_timeouts,
        },
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "summary": {
            "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
            "total_calls": res.model_calls,
            "avg_calls_per_correct": avg_calls,
            "avg_resets": float(res.n_resets.float().mean().item()),
        },
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
    }
    eval_path = checkpoint.replace(".pt", out_suffix)
    with open(eval_path, "w") as f:
        json.dump(out, f, indent=2)

    # Per-puzzle JSONL alongside the summary JSON — clean prefix only.
    eval_jsonl_path = checkpoint.replace(".pt", out_suffix.replace(".json", ".jsonl"))
    with open(eval_jsonl_path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **out}) + "\n")
        for i in prefix_idxs:
            o = outcomes[i]
            is_correct = bool(o["correct"])
            is_wrong = bool(o["wrong"])
            is_timeout = bool(o["timeout"])
            rs = int(o["round_solved"])
            if is_correct:
                forwards_unbatched = (rs + 1) * n_chains
            else:
                forwards_unbatched = max_rounds * n_chains
            row = {
                "kind": "puzzle",
                "puzzle_idx": i,
                "correct": is_correct,
                "wrong": is_wrong,
                "timeout": is_timeout,
                "round_solved": rs,
                "n_resets": int(o["n_resets"]),
                "forwards_unbatched": forwards_unbatched,
            }
            if estimate_sequential:
                seq_v = int(res.forwards_seq[i].item())
                w_idx = int(res.seq_winning_idx[i].item())
                done = int(res.seq_attempts_done[i].item())
                row["forwards_seq"] = seq_v
                row["seq_winning_idx"] = w_idx
                row["seq_attempts_done"] = done
                row["seq_avg_attempt_len"] = (seq_v / max(w_idx + 1, 1)) if w_idx >= 0 else None
            if log_per_round_fill and is_correct:
                row["n_givens"] = res.n_givens[i]
                row["deduction_fills_per_round"] = res.deduction_fills_per_round[i]
                row["decision_fills_per_round"] = res.decision_fills_per_round[i]
                row["deduction_bitflips_per_round"] = res.deduction_bitflips_per_round[i]
                row["decision_bitflips_per_round"] = res.decision_bitflips_per_round[i]
            fh.write(json.dumps(row) + "\n")
    checkpoint_volume.commit()
    print(f"Wrote {eval_path}", flush=True)
    print(f"Wrote {eval_jsonl_path}", flush=True)
    return out


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    n_eval: int = 200,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    log_per_round_fill: bool = False,
    out_suffix: str = ".eval.fixed.json",
    split: str = "test",
    compile: bool = False,
    eval_max_timeouts: int = 50,
):
    result = run.remote(
        checkpoint=checkpoint, n_eval=n_eval,
        threshold=threshold, temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        n_chains=n_chains, batch_size=batch_size, max_rounds=max_rounds,
        augment=augment,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        dropout_p=dropout_p,
        log_per_round_fill=log_per_round_fill,
        eval_max_timeouts=eval_max_timeouts,
        out_suffix=out_suffix,
        split=split,
        compile=compile,
    )
    print(f"\nFinal: {result}", flush=True)
