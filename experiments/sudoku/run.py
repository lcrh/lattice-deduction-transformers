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
    ("eval_max_timeouts",        50),
    ("augment",                  True),
    ("data_augment_digit_perm",  True),
    ("data_augment_dihedral",    True),
    ("use_ema",                  False),
    ("ema_decay",                0.999),
    ("estimate_sequential",      False),
    ("seq_drain_max_rounds",     200),
    ("eval_dropout_p",           0.05),
    ("n_loops",                  16),
    ("num_layers",               4),
    ("dim",                      128),
    ("pre_norm",                 True),
    ("supervise",                "all"),
    ("eval_every",               100),
    ("eval_n_loops",             0),
    ("cell_policy",              "uniform"),
    ("digit_policy",             "softmax"),
    ("backtrack",                "root"),
    ("geometric_p",              0.5),
    ("learn_negation",           False),
    ("snapshot_max_depth",       64),
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
    eval_max_timeouts: int = 50,
    augment: bool = True,
    data_augment_digit_perm: bool = True,
    data_augment_dihedral: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_loops: int = 16,
    num_layers: int = 4,
    dim: int = 128,
    pre_norm: bool = True,
    supervise: str = "all",
    eval_every: int = 100,
    eval_n_loops: int = 0,
    # ---- E2 additions (all default-off / no-op at defaults) ----
    # cell_policy / digit_policy feed BOTH the training-time dpll_step (via
    # step_cfg, so matched-training S2 induces the policy's state distribution)
    # AND the final eval. backtrack / geometric_p / learn_negation feed the
    # eval solve and (for pool-matched training S3-phase-2) the trainer pool.
    cell_policy: str = "uniform",
    digit_policy: str = "softmax",
    backtrack: str = "root",
    geometric_p: float = 0.5,
    learn_negation: bool = False,
    snapshot_max_depth: int = 64,
    ckpt_name: str = "",
    ckpt_subdir: str = "",
    overwrite: bool = False,
    skip_if_done: bool = False,
):
    # Snapshot the call-site arg values BEFORE any local mutation, then dump
    # the config table. (Snapshot locals() outside the comprehension —
    # Python 3 comprehensions have their own scope, so locals() inside one
    # only sees the iter-vars, NOT the function's parameters.)
    _loc_snapshot = dict(locals())
    _arg_values = {name: _loc_snapshot[name] for name, _ in _RUN_PARAMS}
    _print_run_config(_arg_values)

    # Stateful / idempotent guard. When --skip-if-done is set AND we are using
    # a deterministic checkpoint path (--ckpt-name given), a run whose eval.json
    # already landed on the volume is a graceful no-op — we neither retrain nor
    # re-evaluate. This is distinct from --overwrite (which errors on an existing
    # checkpoint); --skip-if-done lets a whole-sweep re-launch execute only the
    # missing (config, seed) pairs. Default-off: plain runs are unaffected.
    # `_resume_eval_only` is set below when --skip-if-done finds a trained
    # checkpoint (.pt) but no eval.json: we then SKIP training and jump straight
    # to eval, loading the existing .pt. This makes a killed-mid-eval run (and
    # the per-puzzle progress-file resume) recoverable without retraining.
    _resume_eval_only = False
    _resume_ckpt_path = None
    if skip_if_done and ckpt_name:
        from pathlib import Path as _Path
        _done_dir = f"{CHECKPOINT_MOUNT}/{ckpt_subdir}" if ckpt_subdir else CHECKPOINT_MOUNT
        _done_eval = _Path(_done_dir) / f"{ckpt_name}.eval.json"
        _done_pt = _Path(_done_dir) / f"{ckpt_name}.pt"
        checkpoint_volume.reload()
        if _done_eval.exists():
            print(f"\n[skip-if-done] {_done_eval} already exists — "
                  f"skipping train + eval (no-op).", flush=True)
            return {"skipped": True, "checkpoint": str(_done_eval.with_suffix("").with_suffix(".pt"))}
        if _done_pt.exists():
            # Trained but not (fully) evaluated — resume at the eval stage.
            print(f"\n[skip-if-done] {_done_pt} exists but no eval.json — "
                  f"skipping training, resuming EVAL only.", flush=True)
            _resume_eval_only = True
            _resume_ckpt_path = _done_pt

    ts = time.strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
        cell_policy=cell_policy,
        digit_policy=digit_policy,
    )
    model_cfg = LoopedTransformerConfig(
        cls_token=conflict_loss_weight > 0,
        n_loops=n_loops,
        num_layers=num_layers,
        dim=dim,
        pre_norm=pre_norm,
    )

    # Checkpoint path routing. When `ckpt_name` is set, a followup launcher
    # requests a deterministic (timestamp-free) path at
    # `{CHECKPOINT_MOUNT}/{ckpt_subdir}/{ckpt_name}.pt` (e.g.
    # /checkpoints/followups/e1/<config>_seed<N>.pt). When empty, everything
    # falls back to the legacy `{CHECKPOINT_MOUNT}/sudoku` + timestamped name.
    if ckpt_name:
        train_out_dir = f"{CHECKPOINT_MOUNT}/{ckpt_subdir}" if ckpt_subdir else CHECKPOINT_MOUNT
        train_name = ckpt_name
        train_no_timestamp = True
    else:
        train_out_dir = f"{CHECKPOINT_MOUNT}/sudoku"
        train_name = f"seed{seed}_{steps}s_bs{batch_size}_aug{int(augment)}_{ts}"
        train_no_timestamp = False

if _resume_eval_only:
        # Trained checkpoint already on the volume — skip training entirely and
        # eval the existing .pt (per-puzzle progress file, if any, resumes below).
        from pathlib import Path as _Path
        ckpt_path = _Path(str(_resume_ckpt_path))
    else:
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
            supervise=supervise,
            eval_every=eval_every,
            # E2 pool-side backtrack matching (default "root" = legacy byte-identical).
            backtrack=backtrack,
            geometric_p=geometric_p,
            learn_negation=learn_negation,
            pool_snapshot_max_depth=snapshot_max_depth,
            model=model_cfg,
            data=SudokuExtremeConfig(
                cache_dir=DATA_MOUNT, batch_size=batch_size, seed=42,
                n_puzzles=n_train_puzzles,
                augment_digit_perm=data_augment_digit_perm,
                augment_dihedral=data_augment_dihedral,
            ),
            out_dir=train_out_dir,
            name=train_name,
            no_timestamp=train_no_timestamp,
            overwrite=overwrite,
        ))
        checkpoint_volume.commit()

    print("\n" + "=" * 60, flush=True)
    print(f"Eval ({n_eval_puzzles} test puzzles)", flush=True)
    print("=" * 60, flush=True)

    ckpt = load_checkpoint(str(ckpt_path))
    # Eval-time loop override (E3-O1). The backbone is a weight-tied loop, so a
    # checkpoint trained at n_loops evals at any n_loops by rebuilding the
    # config with a different n_loops BEFORE load_state_dict — the state dict
    # shape is loop-invariant. Default 0 = keep the checkpoint's native
    # n_loops (rebuilt config byte-identical to the saved one; sanity gate:
    # --eval-n-loops <native> reproduces the no-flag result).
    _saved_model_cfg = dict(ckpt["model_cfg"])
    if eval_n_loops and eval_n_loops > 0:
        _native = _saved_model_cfg.get("n_loops")
        _saved_model_cfg["n_loops"] = eval_n_loops
        print(f"  [eval-n-loops] overriding n_loops {_native} -> {eval_n_loops} "
              f"(weight-tied loop; state dict unchanged)", flush=True)
    cfg_loaded = LoopedTransformerConfig(**_saved_model_cfg)
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

    # ----- Eval early-abort + per-puzzle resume wiring -----
    # Progress file lives next to the checkpoint (.pt -> .eval.progress.jsonl).
    # It is a streaming, append-only per-puzzle log so an interrupted eval can
    # resume without re-solving finished puzzles. All of this is default-off:
    # with eval_max_timeouts<=0 and no pre-existing progress file, the abort
    # never fires, already_done is empty, and behavior is identical to before.
    from pathlib import Path as _Path
    progress_path = ckpt_path.with_suffix(".eval.progress.jsonl")
    _max_to = eval_max_timeouts if eval_max_timeouts > 0 else None

    # Resume: read any existing progress file, build already_done + prior counts,
    # and keep the prior rows for the clean-prefix computation + final jsonl.
    prior_rows: list[dict] = []
    already_done: set = set()
    if progress_path.exists():
        with progress_path.open("r") as _pf:
            for _line in _pf:
                _line = _line.strip()
                if not _line:
                    continue
                _row = json.loads(_line)
                prior_rows.append(_row)
                already_done.add(int(_row["idx"]))
        print(f"  [resume] loaded {len(prior_rows)} prior puzzle outcomes from "
              f"{progress_path.name} ({len(already_done)} indices already done)",
              flush=True)

    solve_cfg = SolveConfig(
        step=eval_step_cfg, max_rounds=eval_max_rounds,
        n_chains=eval_n_chains, batch_size=eval_batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
eval_max_timeouts=_max_to,
        already_done=(already_done or None),
        backtrack=backtrack,
        geometric_p=geometric_p,
        learn_negation=learn_negation,
        snapshot_max_depth=snapshot_max_depth,
    )

    # Streaming callback: append+flush each puzzle as it completes. No per-puzzle
    # volume commit — the file rides along on the single commit after solve().
    _progress_fh = progress_path.open("a")

    def _on_puzzle_done(row: dict) -> None:
        _progress_fh.write(json.dumps(row) + "\n")
        _progress_fh.flush()

    solve_cfg.on_puzzle_done = _on_puzzle_done
    try:
        res = solve(model, x, y, given_mask, solve_cfg)
    finally:
        _progress_fh.close()

    # Merge prior rows (resume) + this run's evicted puzzles into a single
    # outcomes map, then compute the MAXIMAL GAP-FREE PREFIX [0..k]. Reporting
    # over the prefix keeps the pass rate an unbiased sample: fast puzzles that
    # finished past an abort point don't skew the denominator upward.
    outcomes: dict[int, dict] = {}
    for _row in prior_rows:
        outcomes[int(_row["idx"])] = _row
    P_total = res.solved.shape[0]
    for i in range(P_total):
        if i in already_done:
            continue
        # A puzzle actually filled+evicted this run has a real outcome. On a full
        # run that's every index; on an aborted run only a prefix + stragglers.
        if int(res.puzzle_calls[i].item()) < 0:
            continue  # never filled — no outcome
        outcomes[i] = {
            "idx": i,
            "correct": bool(res.correct[i].item()),
            "wrong": bool(res.wrong[i].item()),
            "timeout": bool(res.timeouts[i].item()),
            "round_solved": int(res.round_solved[i].item()),
            "puzzle_calls": int(res.puzzle_calls[i].item()),
        }

    k = -1
    while (k + 1) in outcomes:
        k += 1
    prefix_idxs = range(0, k + 1)

    n = k + 1
    n_correct = sum(1 for i in prefix_idxs if outcomes[i]["correct"])
    n_wrong = sum(1 for i in prefix_idxs if outcomes[i]["wrong"])
    n_timeout = sum(1 for i in prefix_idxs if outcomes[i]["timeout"])
    if res.aborted or already_done:
        print(f"  [prefix] gap-free prefix length n={n} "
              f"(aborted={res.aborted}, resumed={bool(already_done)}, "
              f"outcomes={len(outcomes)}/{P_total})", flush=True)
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
        "eval_aborted": res.aborted,
        "n_evaluated_prefix": n,
        "eval_max_timeouts": eval_max_timeouts,
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
            # E2 backtracking diagnostics (nonzero only for non-root policies).
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
        # Dump only the clean-prefix puzzles, sourcing per-puzzle outcome from
        # the merged `outcomes` map (so resumed indices carry their prior row,
        # not the False/-1 placeholders in this run's `res`).
        for i in prefix_idxs:
            o = outcomes[i]
            is_correct = bool(o["correct"])
            is_wrong = bool(o["wrong"])
            is_timeout = bool(o["timeout"])
            rs = int(o["round_solved"])
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
                "puzzle_calls": int(o["puzzle_calls"]),
                "forwards_unbatched": forwards_unbatched,
            }) + "\n")
    checkpoint_volume.commit()

    # The progress file has served its purpose now that eval.json + eval.jsonl
    # are written and committed. Delete it so a future --skip-if-done resume
    # doesn't mistake it for partial work. Guard with an existence check.
    if progress_path.exists():
        progress_path.unlink()
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
    eval_max_timeouts: int = 50,
    augment: bool = True,
    data_augment_digit_perm: bool = True,
    data_augment_dihedral: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_loops: int = 16,
    num_layers: int = 4,
    dim: int = 128,
    pre_norm: bool = True,
    supervise: str = "all",
    eval_every: int = 100,
    eval_n_loops: int = 0,
    cell_policy: str = "uniform",
    digit_policy: str = "softmax",
    backtrack: str = "root",
    geometric_p: float = 0.5,
    learn_negation: bool = False,
    snapshot_max_depth: int = 64,
    ckpt_name: str = "",
    ckpt_subdir: str = "",
    overwrite: bool = False,
    skip_if_done: bool = False,
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
        eval_max_timeouts=eval_max_timeouts,
        augment=augment,
        data_augment_digit_perm=data_augment_digit_perm,
        data_augment_dihedral=data_augment_dihedral,
        use_ema=use_ema,
        ema_decay=ema_decay,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        eval_dropout_p=eval_dropout_p,
        n_loops=n_loops,
        num_layers=num_layers,
        dim=dim,
        pre_norm=pre_norm,
        supervise=supervise,
        eval_every=eval_every,
        eval_n_loops=eval_n_loops,
        cell_policy=cell_policy,
        digit_policy=digit_policy,
        backtrack=backtrack,
        geometric_p=geometric_p,
        learn_negation=learn_negation,
        snapshot_max_depth=snapshot_max_depth,
        ckpt_name=ckpt_name,
        ckpt_subdir=ckpt_subdir,
        overwrite=overwrite,
        skip_if_done=skip_if_done,
    )
    print(f"\nFinal: {result}", flush=True)
