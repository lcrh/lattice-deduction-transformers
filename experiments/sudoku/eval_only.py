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
from lattice_diffusion.models.transformer2d import _RelBiasAttention
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve
from experiments.sudoku.union_frames import UnionFrames


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
    search: str = "restart",
    use_ban_set: bool = True,
    policy_mode: str = "random",
    union_frames: int = 1,
    union_agg: str = "union",
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
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
            elif isinstance(m, _RelBiasAttention):
                m.attn_dropout_p = dropout_p
                n_mha += 1
        model.train()
        print(f"  Dropout-noise eval: overrode {n_drop} nn.Dropout + "
              f"{n_mha} MHA-internal-attn-dropout layers to p={dropout_p}, "
              f"model in train() mode", flush=True)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if union_frames > 1:
        # CoLT H2: test-time union ensembling over K digit-permutation
        # frames (no retraining). See experiments/sudoku/union_frames.py.
        model = UnionFrames(
            model, n_frames=union_frames, agg=union_agg,
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"  UnionFrames eval wrapper: K={union_frames} agg={union_agg}",
              flush=True)

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
        policy_mode=policy_mode,
    )
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=max_rounds,
        n_chains=n_chains, batch_size=batch_size,
        search=search, use_ban_set=use_ban_set,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        log_per_round_fill=log_per_round_fill,
    )
    print(f"Solving {n} puzzles | n_chains={n_chains} batch_size={batch_size} "
          f"(M={batch_size//n_chains} puzzles/forward) max_rounds={max_rounds}",
          flush=True)
    print(f"  step: threshold={threshold} (deterministic) "
          f"temp_dec={temp_decide} cls_threshold={cls_threshold} "
          f"augment={augment} search={search} policy={policy_mode}", flush=True)

    t0 = time.time()
    res = solve(model, x, y, given_mask, solve_cfg)
    elapsed = time.time() - t0

    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    avg_calls = res.model_calls / max(n_correct, 1)

    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    total_suppressed = int(res.n_suppressed.sum().item())
    total_backjumps = int(res.n_backjumps.sum().item())

    print(f"\n{'='*60}\nRESULT (streaming-queue solver, {n} puzzles)\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}", flush=True)
    print(f"  total_calls={res.model_calls}  avg/correct={avg_calls:.1f}", flush=True)
    if search == "dfs":
        print(f"  DFS: suppressed_wrong_completions={total_suppressed}  "
              f"backjumps={total_backjumps}", flush=True)
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
            "search": search, "use_ban_set": use_ban_set,
            "policy_mode": policy_mode,
            "union_frames": union_frames, "union_agg": union_agg,
        },
        "summary": {
            "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
            "total_calls": res.model_calls,
            "avg_calls_per_correct": avg_calls,
            "avg_resets": float(res.n_resets.float().mean().item()),
            "suppressed_wrong_completions": total_suppressed,
            "backjumps": total_backjumps,
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
                "n_suppressed": int(res.n_suppressed[i].item()),
                "n_backjumps": int(res.n_backjumps[i].item()),
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
    search: str = "restart",
    use_ban_set: bool = True,
    policy_mode: str = "random",
    union_frames: int = 1,
    union_agg: str = "union",
    seed: int = 0,
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
        out_suffix=out_suffix,
        split=split,
        compile=compile,
        search=search,
        use_ban_set=use_ban_set,
        policy_mode=policy_mode,
        union_frames=union_frames,
        union_agg=union_agg,
        seed=seed,
    )
    print(f"\nFinal: {result}", flush=True)
