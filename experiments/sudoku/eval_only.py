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
    # ---- E3 additions (all default-off / no-op at defaults) ----
    eval_n_loops: int = 0,        # 0 = checkpoint's native n_loops (E3-O1)
    deduce_passes: int = 1,       # 1 = legacy single pass (E3-O3)
    deduce_pass_cap: int = 16,    # safety cap for deduce_passes=0 fixpoint mode
    # ---- E2 additions (all default-off / no-op at defaults) ----
    cell_policy: str = "uniform",   # E2-S1 cell selection: uniform|mrv|min_entropy|max_entropy
    digit_policy: str = "softmax",  # E2-S1 digit selection: softmax|argmax|rank_k
    backtrack: str = "root",        # E2-S3 backtracking: root|last|geometric|uniform_depth|last+negate
    geometric_p: float = 0.5,       # p for backtrack=geometric
    learn_negation: bool = False,   # force negation on (implied by backtrack=last+negate)
    snapshot_max_depth: int = 64,   # per-chain decision snapshot stack depth
    ckpt_name: str = "",          # deterministic output routing (mirrors run.py)
    ckpt_subdir: str = "",        # -> {CHECKPOINT_MOUNT}/{ckpt_subdir}/{ckpt_name}.eval.json
    skip_if_done: bool = False,   # idempotent skip if the eval.json already landed
):
    # Deterministic output routing (mirrors run.py). When `ckpt_name` is set,
    # eval artifacts land at `{CHECKPOINT_MOUNT}/{ckpt_subdir}/{ckpt_name}.eval.json`
    # (E3's `<evalconfig>__on__<input>_seed<N>` naming under followups/e3) —
    # decoupled from the (read-only, shared E1) input checkpoint path. When
    # empty, artifacts fall back next to the input checkpoint (legacy default).
    from pathlib import Path as _Path
    if ckpt_name:
        # Routed (followup) output MUST use the exchange-contract suffix
        # `.eval.json` — collect.py, configs.py status/remaining, and
        # `_common.volume_done_set` all key off `.eval.json`. The standalone
        # `out_suffix` default (`.eval.fixed.json`) is for un-routed one-off
        # evals only; honoring it here would make every followup eval invisible
        # to the whole collect/status pipeline.
        _routed_suffix = ".eval.json"
        _out_dir = f"{CHECKPOINT_MOUNT}/{ckpt_subdir}" if ckpt_subdir else CHECKPOINT_MOUNT
        _out_base = f"{_out_dir}/{ckpt_name}"
        eval_path = f"{_out_base}{_routed_suffix}"
        eval_jsonl_path = f"{_out_base}{_routed_suffix.replace('.json', '.jsonl')}"
    else:
        eval_path = checkpoint.replace(".pt", out_suffix)
        eval_jsonl_path = checkpoint.replace(".pt", out_suffix.replace(".json", ".jsonl"))

    # Idempotent skip (mirrors run.py --skip-if-done). Default-off: only fires
    # when both --skip-if-done and --ckpt-name are given and the eval.json is
    # already on the volume, so whole-sweep re-launches touch only missing runs.
    if skip_if_done and ckpt_name:
        checkpoint_volume.reload()
        if _Path(eval_path).exists():
            print(f"\n[skip-if-done] {eval_path} already exists — "
                  f"skipping eval (no-op).", flush=True)
            return {"skipped": True, "eval_path": eval_path}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print("RUN CONFIG  (eval_only)", flush=True)
    print(f"  checkpoint={checkpoint}", flush=True)
    print(f"  n_eval={n_eval} threshold={threshold} cls_threshold={cls_threshold} "
          f"eval_max_timeouts={eval_max_timeouts} out_suffix={out_suffix}",
          flush=True)
    print(f"Loading: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    # Eval-time loop override (E3-O1). The backbone is a weight-tied loop, so a
    # checkpoint trained at any L_train evals at any L_eval by rebuilding the
    # config with a different n_loops BEFORE load_state_dict — the state dict
    # shape is loop-invariant. Default 0 = keep the checkpoint's native
    # n_loops, making the rebuilt config byte-identical to the saved one
    # (sanity gate: --eval-n-loops <native> reproduces the no-flag result).
    saved_cfg = dict(ckpt["model_cfg"])
    native_loops = saved_cfg.get("n_loops")
    if eval_n_loops and eval_n_loops > 0:
        saved_cfg["n_loops"] = eval_n_loops
        print(f"  [eval-n-loops] overriding n_loops {native_loops} -> "
              f"{eval_n_loops} (weight-tied loop; state dict unchanged)",
              flush=True)
    cfg = LoopedTransformerConfig(**saved_cfg)
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
        deduce_passes=deduce_passes,
        deduce_pass_cap=deduce_pass_cap,
        cell_policy=cell_policy,
        digit_policy=digit_policy,
    )
    _max_to = eval_max_timeouts if eval_max_timeouts > 0 else None
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=max_rounds,
        n_chains=n_chains, batch_size=batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        log_per_round_fill=log_per_round_fill,
        eval_max_timeouts=_max_to,
        backtrack=backtrack,
        geometric_p=geometric_p,
        learn_negation=learn_negation,
        snapshot_max_depth=snapshot_max_depth,
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
    if res.per_pass_deduced_total:
        # E3-O3: per-pass unsound compounding (multi-pass deduce only).
        pp = [
            (i, d, u, (u / d if d else 0.0))
            for i, (d, u) in enumerate(
                zip(res.per_pass_deduced_total, res.per_pass_unsound_total))
        ]
        print(f"  Per-pass unsound (deduce_passes={deduce_passes}):", flush=True)
        for i, d, u, r in pp:
            print(f"    pass {i}: {u} unsound / {d} deduced  (rate={r:.4%})",
                  flush=True)
    if backtrack != "root":
        un_rate = res.n_unsound_negations / max(res.n_negations, 1)
        cd_mean = (sum(res.conflict_depths) / len(res.conflict_depths)
                   if res.conflict_depths else 0.0)
        print(f"  Backtrack={backtrack}: {len(res.conflict_depths)} conflicts "
              f"(mean depth {cd_mean:.1f})  "
              f"negations={res.n_negations} unsound={res.n_unsound_negations} "
              f"(rate={un_rate:.4%})", flush=True)
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
            # E3 knobs recorded for downstream collect/plot.
            "eval_n_loops": eval_n_loops,
            "native_n_loops": native_loops,
            "effective_n_loops": (eval_n_loops if eval_n_loops and eval_n_loops > 0
                                  else native_loops),
            "deduce_passes": deduce_passes,
            "deduce_pass_cap": deduce_pass_cap,
            # E2 search-process knobs recorded for downstream collect/plot.
            "cell_policy": cell_policy,
            "digit_policy": digit_policy,
            "backtrack": backtrack,
            "geometric_p": geometric_p,
            "learn_negation": learn_negation,
        },
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "summary": {
            "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
            "total_calls": res.model_calls,
            "avg_calls_per_correct": avg_calls,
            # Prefix-scoped means: aborted/resume runs must not average over
            # never-filled (-1) or post-gap straggler puzzles.
            "avg_resets": (
                float(sum(outcomes[i]["n_resets"] for i in prefix_idxs) / n_prefix)
                if n_prefix > 0 else 0.0),
            # E2 sequential-cost + puzzle-calls summary (README key-cost metric).
            "avg_puzzle_calls": (
                float(sum(outcomes[i]["puzzle_calls"] for i in prefix_idxs) / n_prefix)
                if n_prefix > 0 else -1.0),
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
            # E3-O3 per-pass-index compounding (deduce-to-fixpoint). Lists
            # indexed by deduce pass number; `unsound[i]` bits killed a
            # GT-alive bit on pass i, `deduced[i]` total bits killed on pass i.
            # Inert (empty lists) for single-pass evals (deduce_passes==1).
            "per_pass": {
                "deduced": res.per_pass_deduced_total,
                "unsound": res.per_pass_unsound_total,
            },
            # E2 backtracking diagnostics.
            "backtrack_policy": res.backtrack_policy,
            "n_negations": res.n_negations,
            "n_unsound_negations": res.n_unsound_negations,
            "unsound_negation_rate": (
                res.n_unsound_negations / max(res.n_negations, 1)),
            "n_conflicts_recorded": len(res.conflict_depths),
            # Histograms (per-conflict lists) — capped in the jsonl, summarized here.
            "conflict_depth_mean": (
                sum(res.conflict_depths) / len(res.conflict_depths)
                if res.conflict_depths else 0.0),
            "backtrack_target_mean": (
                sum(res.backtrack_targets) / len(res.backtrack_targets)
                if res.backtrack_targets else 0.0),
        },
        # Full per-conflict histograms (decision-depth-at-conflict + target).
        # Only nonempty for non-root backtrack policies; safe to store (a few
        # thousand small ints at most).
        "conflict_depths": res.conflict_depths,
        "backtrack_targets": res.backtrack_targets,
    }
    # eval_path / eval_jsonl_path were resolved up-front (deterministic output
    # routing under --ckpt-name, else next to the input checkpoint).
    _Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(out, f, indent=2)

# Per-puzzle JSONL alongside the summary JSON — clean prefix only.
    # `forwards_unbatched` = "model forwards this puzzle would cost if we
    # ran sequentially with no slot-batching and no chain-batching" =
    # K * (round_solved + 1) for solved (0-round-solve = 1 forward; ×K
    # because all K chains run, each as its own forward in sequential mode).
    # Wrongs and timeouts are charged the full K * max_rounds — a wrong
    # confident answer is no more useful than a timeout.
    # NB: with deduce_passes != 1 each round costs >1 forward, so this
    # rounds-based estimate under-counts; `total_calls` (model_calls) is the
    # honest per-forward cost. forwards_unbatched stays rounds-based for
    # continuity with the single-pass reports.
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
    eval_n_loops: int = 0,
    deduce_passes: int = 1,
    deduce_pass_cap: int = 16,
    cell_policy: str = "uniform",
    digit_policy: str = "softmax",
    backtrack: str = "root",
    geometric_p: float = 0.5,
    learn_negation: bool = False,
    snapshot_max_depth: int = 64,
    ckpt_name: str = "",
    ckpt_subdir: str = "",
    skip_if_done: bool = False,
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
        eval_n_loops=eval_n_loops,
        deduce_passes=deduce_passes,
        deduce_pass_cap=deduce_pass_cap,
        cell_policy=cell_policy,
        digit_policy=digit_policy,
        backtrack=backtrack,
        geometric_p=geometric_p,
        learn_negation=learn_negation,
        snapshot_max_depth=snapshot_max_depth,
        ckpt_name=ckpt_name,
        ckpt_subdir=ckpt_subdir,
        skip_if_done=skip_if_done,
    )
    print(f"\nFinal: {result}", flush=True)
