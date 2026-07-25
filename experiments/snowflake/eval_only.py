"""Eval an existing checkpoint with the hybrid-batched solver.

Eval data: seed=200, zero_hint_weight=1.0 (all SAT), n=200 (post-filter).

Usage:
    uv run modal run --detach experiments/snowflake/eval_only.py \
      --checkpoint /checkpoints/snowflake/seed0_1000s_bs512_aug1_<ts>.pt \
      --temp-eliminate 0.0
"""

import json
import time

import modal
import torch
from torch import nn

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve
from experiments.snowflake.data import SnowflakeConfig, SnowflakeDataset
from experiments.snowflake.train import (
    N_CHANNELS, VOCAB, _build_state, _given_mask,
)


app = modal.App("snowflake-eval-only")


@app.function(
    image=image, gpu="B200", timeout=3600,
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
    out_suffix: str = ".eval.fixed.json",
    data_suffix: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    cfg = LoopedTransformerConfig(**ckpt["model_cfg"])
    constraint_group_vocab = cfg.n_channels - N_CHANNELS
    if constraint_group_vocab < 0:
        raise ValueError(
            f"checkpoint n_channels={cfg.n_channels} is smaller than "
            f"Snowflake's legacy width {N_CHANNELS}"
        )
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
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

    eval_ds = SnowflakeDataset(SnowflakeConfig(
        data_path=f"{DATA_MOUNT}/snowflake_test{data_suffix}.parquet",
        n_puzzles=n_eval, batch_size=n_eval, seed=200,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        constraint_group_vocab=constraint_group_vocab,
    ))
    eval_batch = eval_ds.next_batch(); eval_ds.close()
    if constraint_group_vocab > 0:
        x_all, y_all, mask_all, sat, groups_all = eval_batch
    else:
        x_all, y_all, mask_all, sat = eval_batch
        groups_all = None
    sat_mask = sat.bool()
    x = x_all[sat_mask].to(device).float()
    y = y_all[sat_mask].to(device).float()
    in_puzzle_mask = mask_all[sat_mask].to(device).bool()
    group_features = (
        groups_all[sat_mask].to(device).float()
        if groups_all is not None else None
    )
    state = _build_state(x, in_puzzle_mask, group_features)
    given_mask = _given_mask(x)
    n = x.shape[0]
    print(f"Loaded {n}/{n_eval} SAT snowflake eval puzzles", flush=True)

    # Eval uses deterministic threshold elimination (no stochastic temp).
    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
        augment_dihedral=False,    # snowflake: digit-perm only
        vocab_dim=VOCAB,
    )
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=max_rounds,
        n_chains=n_chains, batch_size=batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
    )
    print(f"Solving {n} puzzles | n_chains={n_chains} batch_size={batch_size} "
          f"(M={batch_size//n_chains} puzzles/forward) max_rounds={max_rounds}",
          flush=True)
    print(f"  step: threshold={threshold} (deterministic) "
          f"temp_dec={temp_decide} cls_threshold={cls_threshold} "
          f"augment={augment}", flush=True)

    t0 = time.time()
    res = solve(model, state, y, given_mask, solve_cfg, in_puzzle_mask=in_puzzle_mask)
    elapsed = time.time() - t0

    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    avg_calls = res.model_calls / max(n_correct, 1)

    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    print(f"\n{'='*60}\nRESULT (streaming-queue solver, {n} puzzles)\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}", flush=True)
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
        "n_eval": n,
        "solver": "hybrid_per_chunk",
        "solver_config": {
            "threshold": threshold, "temp_decide": temp_decide,
            "cls_threshold": cls_threshold,
            "n_chains": n_chains, "batch_size": batch_size,
            "max_rounds": max_rounds,
            "augment": augment,
        },
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

    # Per-puzzle JSONL alongside the summary JSON.
    # `forwards_unbatched` = "model forwards this puzzle would cost if we
    # ran sequentially with no slot-batching and no chain-batching" =
    # K * (round_solved + 1) for solved (0-round-solve = 1 forward; ×K
    # because all K chains run, each as its own forward in sequential mode).
    # Wrongs and timeouts are charged the full K * max_rounds — a wrong
    # confident answer is no more useful than a timeout.
    eval_jsonl_path = checkpoint.replace(".pt", out_suffix.replace(".json", ".jsonl"))
    with open(eval_jsonl_path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **out}) + "\n")
        for i in range(n):
            is_correct = bool(res.correct[i].item())
            is_wrong = bool(res.wrong[i].item())
            is_timeout = bool(res.timeouts[i].item())
            rs = int(res.round_solved[i].item())
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
                "n_resets": int(res.n_resets[i].item()),
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
    out_suffix: str = ".eval.fixed.json",
    data_suffix: str = "",
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
        out_suffix=out_suffix,
        data_suffix=data_suffix,
    )
    print(f"\nFinal: {result}", flush=True)
