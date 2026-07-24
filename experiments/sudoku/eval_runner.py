"""Shared eval orchestration for baseline and optional Modal entrypoints.

Loads a checkpoint, optionally overrides n_loops, builds StepConfig/SolveConfig,
runs solve(), and writes clean-prefix eval artifacts. Followups inject strategies
onto the configs before calling `run_eval`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig, SudokuExtremeDataset
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import CHECKPOINT_MOUNT, DATA_MOUNT, checkpoint_volume
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.hooks import public_asdict
from experiments.sudoku.reporting import prefix_outcomes, prefix_summary_metrics
from experiments.sudoku.solve import SolveConfig, SolveResult, solve


@dataclass
class EvalRunSpec:
    checkpoint: str
    n_eval: int = 200
    threshold: float = 0.10
    temp_decide: float = 1.5
    cls_threshold: float = 0.6
    n_chains: int = 64
    batch_size: int = 512
    max_rounds: int = 1000
    augment: bool = True
    # "auto" recovers the training policy from checkpoint metadata. Explicit
    # zero_z provides the E6 eval-time zero-carry probe.
    carry_latent: str = "auto"
    estimate_sequential: bool = False
    seq_drain_max_rounds: int = 200
    dropout_p: float = 0.05
    log_per_round_fill: bool = False
    out_suffix: str = ".eval.fixed.json"
    split: str = "test"
    compile: bool = False
    eval_max_timeouts: int = 50
    # Optional loop override (0 = keep checkpoint native).
    eval_n_loops: int = 0
    # Artifact routing (empty => next to checkpoint).
    ckpt_name: str = ""
    ckpt_subdir: str = ""
    skip_if_done: bool = False
    # Extra solver_config / diag keys contributed by callers.
    extra_solver_config: dict[str, Any] = field(default_factory=dict)
    enrich_result: Callable[[dict, SolveResult, dict], None] | None = None
    # Pre-built configs (callers attach strategies here).
    step_cfg: StepConfig | None = None
    solve_cfg: SolveConfig | None = None


def resolve_eval_paths(spec: EvalRunSpec) -> tuple[str, str]:
    if spec.ckpt_name:
        # Routed deterministic output always uses `.eval.json` so collect/status
        # pipelines find artifacts (standalone `out_suffix` is for one-offs).
        suffix = ".eval.json"
        out_dir = (f"{CHECKPOINT_MOUNT}/{spec.ckpt_subdir}"
                   if spec.ckpt_subdir else CHECKPOINT_MOUNT)
        base = f"{out_dir}/{spec.ckpt_name}"
        return f"{base}{suffix}", f"{base}{suffix.replace('.json', '.jsonl')}"
    eval_path = spec.checkpoint.replace(".pt", spec.out_suffix)
    return eval_path, eval_path.replace(".json", ".jsonl")


def run_eval(spec: EvalRunSpec) -> dict:
    eval_path, eval_jsonl_path = resolve_eval_paths(spec)
    if spec.skip_if_done and spec.ckpt_name:
        checkpoint_volume.reload()
        if Path(eval_path).exists():
            print(f"\n[skip-if-done] {eval_path} already exists — skipping eval.",
                  flush=True)
            return {"skipped": True, "eval_path": eval_path}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print("RUN CONFIG  (eval)", flush=True)
    print(f"  checkpoint={spec.checkpoint}", flush=True)
    print(f"  n_eval={spec.n_eval} threshold={spec.threshold} "
          f"cls_threshold={spec.cls_threshold} "
          f"eval_max_timeouts={spec.eval_max_timeouts} out_suffix={spec.out_suffix}",
          flush=True)

    print(f"Loading: {spec.checkpoint}", flush=True)
    ckpt = load_checkpoint(spec.checkpoint)
    saved_cfg = dict(ckpt["model_cfg"])
    native_loops = saved_cfg.get("n_loops")
    if spec.eval_n_loops and spec.eval_n_loops > 0:
        saved_cfg["n_loops"] = spec.eval_n_loops
        print(f"  [eval-n-loops] overriding n_loops {native_loops} -> "
              f"{spec.eval_n_loops}", flush=True)
    cfg = LoopedTransformerConfig(**saved_cfg)
    saved_policy = (
        ckpt.get("train_cfg", {}).get("step", {}).get("carry_latent", "off")
    )
    carry_policy = saved_policy if spec.carry_latent == "auto" else spec.carry_latent
    expected_mode = {"off": "off", "h": "h", "z": "z", "zero_z": "z"}.get(
        carry_policy
    )
    if expected_mode is None or expected_mode != cfg.carry_mode:
        raise ValueError(
            f"eval carry policy {carry_policy!r} is incompatible with "
            f"checkpoint carry_mode {cfg.carry_mode!r}"
        )
    if carry_policy != "off" and spec.augment:
        raise ValueError(
            "carry checkpoint eval requires augment=False (fixed latent frame)"
        )
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if spec.compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)
    if spec.dropout_p > 0.0:
        n_drop = n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = spec.dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = spec.dropout_p
                n_mha += 1
        model.train()
        print(f"  Dropout-noise eval: overrode {n_drop} nn.Dropout + "
              f"{n_mha} MHA layers to p={spec.dropout_p}", flush=True)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    eval_ds = SudokuExtremeDataset(SudokuExtremeConfig(
        cache_dir=DATA_MOUNT, split=spec.split, n_puzzles=spec.n_eval,
        batch_size=spec.n_eval, seed=200,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        augment_digit_perm=False, augment_dihedral=False,
    ))
    x_all, y_all, sat = eval_ds.next_batch(); eval_ds.close()
    sat_mask = sat.bool()
    x = x_all[sat_mask].to(device).float()
    y = y_all[sat_mask].to(device).float()
    given_mask = (x.sum(dim=-1) == 1)
    n = x.shape[0]
    print(f"Loaded {n}/{spec.n_eval} SAT eval puzzles", flush=True)

    step_cfg = spec.step_cfg or StepConfig(
        threshold=spec.threshold,
        temp_decide=spec.temp_decide,
        cls_threshold=spec.cls_threshold,
        augment=spec.augment,
        carry_latent=carry_policy,
    )
    if step_cfg.carry_latent != carry_policy:
        raise ValueError(
            f"step_cfg carry policy {step_cfg.carry_latent!r} does not match "
            f"requested eval policy {carry_policy!r}"
        )
    _max_to = spec.eval_max_timeouts if spec.eval_max_timeouts > 0 else None
    solve_cfg = spec.solve_cfg or SolveConfig(
        step=step_cfg, max_rounds=spec.max_rounds,
        n_chains=spec.n_chains, batch_size=spec.batch_size,
        estimate_sequential=spec.estimate_sequential,
        seq_drain_max_rounds=spec.seq_drain_max_rounds,
        log_per_round_fill=spec.log_per_round_fill,
        eval_max_timeouts=_max_to,
    )
    if solve_cfg.step is None:
        solve_cfg.step = step_cfg

    print(f"Solving {n} puzzles | n_chains={spec.n_chains} "
          f"batch_size={spec.batch_size} max_rounds={spec.max_rounds} "
          f"eval_max_timeouts={_max_to}", flush=True)

    import time
    t0 = time.time()
    res = solve(model, x, y, given_mask, solve_cfg)
    elapsed = time.time() - t0

    outcomes, prefix_idxs, n_prefix = prefix_outcomes(res, n)
    metrics = prefix_summary_metrics(outcomes, prefix_idxs)
    n_correct = metrics["n_correct"]
    n_wrong = metrics["n_wrong"]
    n_timeout = metrics["n_timeout"]
    avg_calls = res.model_calls / max(n_correct, 1)
    if res.aborted:
        print(f"  [prefix] gap-free prefix length n={n_prefix} "
              f"(aborted={res.aborted}, outcomes={len(outcomes)}/{n})",
              flush=True)

    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    print(f"\n{'='*60}\nRESULT (streaming-queue solver, {n_prefix} puzzles)\n{'='*60}",
          flush=True)
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

    effective_loops = (spec.eval_n_loops if spec.eval_n_loops and spec.eval_n_loops > 0
                       else native_loops)
    solver_config = {
        "threshold": spec.threshold, "temp_decide": spec.temp_decide,
        "cls_threshold": spec.cls_threshold,
        "n_chains": spec.n_chains, "batch_size": spec.batch_size,
        "max_rounds": spec.max_rounds,
        "augment": spec.augment,
        "carry_latent": carry_policy,
        "eval_max_timeouts": spec.eval_max_timeouts,
        "eval_n_loops": spec.eval_n_loops,
        "native_n_loops": native_loops,
        "effective_n_loops": effective_loops,
        "step_cfg": public_asdict(step_cfg),
        **spec.extra_solver_config,
    }
    out = {
        "checkpoint": spec.checkpoint,
        "n_eval": n_prefix,
        "n_eval_puzzles": n_prefix,
        "eval_aborted": res.aborted,
        "n_evaluated_prefix": n_prefix,
        "eval_max_timeouts": spec.eval_max_timeouts,
        "solver": "hybrid_per_chunk",
        "solver_config": solver_config,
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "summary": {
            "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
            "total_calls": res.model_calls,
            "avg_calls_per_correct": avg_calls,
            "avg_resets": metrics["avg_resets"],
            "avg_puzzle_calls": metrics["avg_puzzle_calls"],
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
            "per_pass": {
                "deduced": list(res.per_pass_deduced_total),
                "unsound": list(res.per_pass_unsound_total),
            },
            "backtrack_policy": res.backtrack_policy,
            "n_negations": res.n_negations,
            "n_unsound_negations": res.n_unsound_negations,
            "unsound_negation_rate": (
                res.n_unsound_negations / max(res.n_negations, 1)),
            "n_conflicts_recorded": len(res.conflict_depths),
            "conflict_depth_mean": (
                sum(res.conflict_depths) / len(res.conflict_depths)
                if res.conflict_depths else 0.0),
            "backtrack_target_mean": (
                sum(res.backtrack_targets) / len(res.backtrack_targets)
                if res.backtrack_targets else 0.0),
        },
        "conflict_depths": res.conflict_depths,
        "backtrack_targets": res.backtrack_targets,
    }
    if spec.enrich_result is not None:
        spec.enrich_result(out, res, {
            "outcomes": outcomes, "prefix_idxs": prefix_idxs,
            "native_loops": native_loops,
        })

    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(out, f, indent=2)

    with open(eval_jsonl_path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **out}) + "\n")
        for i in prefix_idxs:
            o = outcomes[i]
            is_correct = bool(o["correct"])
            rs = int(o["round_solved"])
            forwards_unbatched = ((rs + 1) * spec.n_chains if is_correct
                                  else spec.max_rounds * spec.n_chains)
            row = {
                "kind": "puzzle",
                "puzzle_idx": i,
                "correct": is_correct,
                "wrong": bool(o["wrong"]),
                "timeout": bool(o["timeout"]),
                "round_solved": rs,
                "n_resets": int(o.get("n_resets", 0)),
                "puzzle_calls": int(o["puzzle_calls"]),
                "forwards_unbatched": forwards_unbatched,
            }
            if spec.estimate_sequential:
                seq_v = int(res.forwards_seq[i].item())
                w_idx = int(res.seq_winning_idx[i].item())
                done = int(res.seq_attempts_done[i].item())
                row["forwards_seq"] = seq_v
                row["seq_winning_idx"] = w_idx
                row["seq_attempts_done"] = done
                row["seq_avg_attempt_len"] = (
                    (seq_v / max(w_idx + 1, 1)) if w_idx >= 0 else None)
            if spec.log_per_round_fill and is_correct:
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
