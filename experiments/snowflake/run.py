"""Modal entry point for snowflake: train then evaluate.

Usage:
    uv run modal run experiments/snowflake/run.py
    uv run modal run experiments/snowflake/run.py --steps 2000 --batch-size 512
"""

import dataclasses
import json
import time
from dataclasses import asdict
from pathlib import Path

import modal
import torch
from torch import nn


# Keep in sync with run() signature defaults. Used for the run-config table
# at the top of every run, with non-defaults highlighted.
_RUN_PARAMS: tuple[tuple[str, object], ...] = (
    ("steps",                    4000),
    ("batch_size",               512),
    ("n_eval_puzzles",           200),
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
    ("use_ema",                  False),
    ("ema_decay",                0.999),
    ("estimate_sequential",      False),
    ("seq_drain_max_rounds",     200),
    ("eval_dropout_p",           0.05),
    ("n_train_puzzles",          None),
    ("train_orders",             ""),
    ("eval_orders",              ""),
    ("translate_aug",            False),
    ("use_rope",                 False),
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
    GRID_COLS, GRID_ROWS, N_CHANNELS, SEQ_LEN, VOCAB,
    TrainConfig, _build_state, _given_mask, train,
)


app = modal.App("snowflake")


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
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_train_puzzles: int | None = None,
    train_orders: str = "",
    eval_orders: str = "",
    translate_aug: bool = False,
    use_rope: bool = False,
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

    def _parse_orders(s: str) -> list[int] | None:
        """Comma-separated order list -> list[int], or None (empty = no filter)."""
        s = (s or "").strip()
        if not s:
            return None
        return [int(tok) for tok in s.split(",") if tok.strip() != ""]

    train_orders_list = _parse_orders(train_orders)
    eval_orders_list = _parse_orders(eval_orders)

    ts = time.strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint path routing. When `ckpt_name` is set, a followup launcher
    # requests a deterministic (timestamp-free) path at
    # `{CHECKPOINT_MOUNT}/{ckpt_subdir}/{ckpt_name}.pt` (e.g.
    # /checkpoints/followups/e4/<config>_seed<N>.pt). When empty, everything
    # falls back to the legacy `{CHECKPOINT_MOUNT}/snowflake` + timestamped
    # name.
    if ckpt_name:
        train_out_dir = f"{CHECKPOINT_MOUNT}/{ckpt_subdir}" if ckpt_subdir else CHECKPOINT_MOUNT
        train_name = ckpt_name
        train_no_timestamp = True
    else:
        train_out_dir = f"{CHECKPOINT_MOUNT}/snowflake"
        train_name = f"seed{seed}_{steps}s_bs{batch_size}_aug{int(augment)}_{ts}"
        train_no_timestamp = False

    # Idempotent re-runs. When `--skip-if-done` is set on a deterministic-path
    # run (`ckpt_name` set), and the eval.json artifact already landed on the
    # volume, skip training+eval entirely and return a no-op result. This lets
    # a whole sweep be re-launched to fill only the missing (config, seed)
    # pairs. Distinct from `--overwrite` (which errors on an existing ckpt).
    # Only meaningful for deterministic paths; a legacy timestamped run always
    # produces a fresh path, so the check is skipped when `ckpt_name` is empty.
    if skip_if_done and ckpt_name:
        done_ckpt_path = Path(train_out_dir) / f"{train_name}.pt"
        done_eval_json_path = Path(train_out_dir) / f"{train_name}.eval.json"
        checkpoint_volume.reload()
        if done_eval_json_path.exists():
            print("=" * 60, flush=True)
            print(f"SKIP: {done_eval_json_path} already exists on the volume; "
                  f"--skip-if-done set, returning without retraining/re-eval.",
                  flush=True)
            print("=" * 60, flush=True)
            return {"skipped": True, "checkpoint": str(done_ckpt_path)}

    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
        augment_dihedral=False,    # snowflake: digit-perm only (covering grid is hex)
        vocab_dim=VOCAB,
    )
    model_cfg = LoopedTransformerConfig(
        n_channels=N_CHANNELS, seq_len=SEQ_LEN,
        grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
        cls_token=conflict_loss_weight > 0,
        use_rope=use_rope,
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
        data=SnowflakeConfig(
            data_path=f"{DATA_MOUNT}/snowflake_train.parquet",
            batch_size=batch_size, seed=42,
            n_puzzles=n_train_puzzles,
            orders=train_orders_list,          # E4: restrict train to these orders
            translate_aug=translate_aug,       # E4: positional-confound mitigation (train only)
        ),
        eval_data=SnowflakeConfig(
            data_path=f"{DATA_MOUNT}/snowflake_test.parquet",
            n_puzzles=200, batch_size=200, seed=200,
            zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
            orders=eval_orders_list,           # E4: restrict in-train eval to these orders
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

    # Eval data: snowflake test split, all SAT (zero_hint_weight=1.0).
    # return_orders=True so we can carry each surviving puzzle's order `n`
    # through the sat_mask and build the per-order breakdown below. The eval
    # dataset uses a single full batch (one next_batch() call) so the returned
    # rows align 1:1 with the solve results.
    #
    # IMPORTANT: cap the batch to the number of UNIQUE puzzles in the filtered
    # pool. The prefetch loop cycles-with-reshuffle when asked for more samples
    # than exist, which would silently pad the eval batch with DUPLICATE puzzles
    # and inflate the per-order denominators / headline correct/n. With an order
    # filter the pool can easily be smaller than n_eval_puzzles, so we build the
    # dataset first (n_puzzles caps its pool), read its true size, and request a
    # single batch of exactly that many unique puzzles. (An empty pool raises in
    # the dataset constructor rather than hanging — see SnowflakeDataset.)
    eval_cfg_probe = SnowflakeConfig(
        data_path=f"{DATA_MOUNT}/snowflake_test.parquet",
        n_puzzles=n_eval_puzzles, batch_size=1, seed=200,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        orders=eval_orders_list,           # E4: restrict eval to these orders
        return_orders=True,
    )
    eval_pool_size = SnowflakeDataset(eval_cfg_probe).n_puzzles
    if eval_pool_size < n_eval_puzzles:
        print(f"  WARNING: only {eval_pool_size} unique eval puzzles available "
              f"(requested {n_eval_puzzles}"
              + (f", orders={eval_orders_list}" if eval_orders_list else "")
              + f"); evaluating on all {eval_pool_size} unique puzzles "
              "(no duplicate padding).", flush=True)
    eval_ds = SnowflakeDataset(dataclasses.replace(
        eval_cfg_probe, batch_size=eval_pool_size,
    ))
    x, y, mask, sat, orders_t = eval_ds.next_batch(); eval_ds.close()
    sat_mask = sat.bool()
    x = x[sat_mask].to(device).float()
    y = y[sat_mask].to(device).float()
    in_puzzle_mask = mask[sat_mask].to(device).bool()
    # Per-puzzle order for the surviving (SAT) eval rows, aligned to solve order.
    eval_orders_per_puzzle = orders_t[sat_mask].tolist()
    state = _build_state(x, in_puzzle_mask)
    given_mask = _given_mask(x)
    n_sat = x.shape[0]
    print(f"  Loaded {n_sat}/{eval_pool_size} SAT snowflake eval puzzles "
          f"(unique pool; requested {n_eval_puzzles})", flush=True)

    # Build a separate eval-time step_cfg that may use a different
    # cls_threshold than training (snowflake inherits the cls=0.5/0.6
    # decoupling from sudoku). Mask + dihedral flags carry over.
    eval_step_cfg = dataclasses.replace(step_cfg, cls_threshold=eval_cls_threshold)
    solve_cfg = SolveConfig(
        step=eval_step_cfg, max_rounds=eval_max_rounds,
        n_chains=eval_n_chains, batch_size=eval_batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
    )
    res = solve(model, state, y, given_mask, solve_cfg, in_puzzle_mask=in_puzzle_mask)

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

    # ---- E4 per-order breakdown ----
    # Group per-puzzle solve outcomes by the puzzle's order `n`. `res` fields
    # (correct/wrong/timeouts/puzzle_calls) are indexed by puzzle position i,
    # which aligns 1:1 with `eval_orders_per_puzzle[i]` (both derived from the
    # same sat_mask-filtered single eval batch, in the same row order).
    per_order: dict[str, dict] = {}
    for i in range(n):
        o = int(eval_orders_per_puzzle[i])
        key = str(o)
        bucket = per_order.setdefault(
            key,
            {"correct": 0, "wrong": 0, "timeout": 0,
             "calls": 0, "calls_n": 0, "n": 0},
        )
        bucket["n"] += 1
        if bool(res.correct[i].item()):
            bucket["correct"] += 1
        if bool(res.wrong[i].item()):
            bucket["wrong"] += 1
        if bool(res.timeouts[i].item()):
            bucket["timeout"] += 1
        # `puzzle_calls` is -1 for never-scheduled puzzles; only real counts
        # contribute. `calls_n` records how many puzzles that was, so a
        # downstream calls/solve normalizes by the right denominator rather
        # than treating skipped puzzles as zero-cost.
        pc = int(res.puzzle_calls[i].item())
        if pc > 0:
            bucket["calls"] += pc
            bucket["calls_n"] += 1
    # Print the per-order table.
    print(f"\n{'='*60}\nPER-ORDER BREAKDOWN\n{'='*60}", flush=True)
    for key in sorted(per_order, key=lambda k: int(k)):
        b = per_order[key]
        print(f"  order {key}: correct={b['correct']}/{b['n']}  "
              f"wrong={b['wrong']}  timeout={b['timeout']}  calls={b['calls']}",
              flush=True)
    print(f"{'='*60}", flush=True)

    eval_json_path = ckpt_path.with_suffix(".eval.json")
    eval_json_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "n_eval_puzzles": n,
        "per_order": per_order,
        "n_chains": res.n_chains,
        "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
        "model_calls_total": res.model_calls,
        "avg_rounds_solved": avg_rounds_solved,
        "avg_resets": avg_resets,
        "step_cfg": asdict(step_cfg),
        "max_rounds": eval_max_rounds,
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
    with eval_jsonl_path.open("w") as fh:
        fh.write(json.dumps({
            "kind": "header",
            "checkpoint": str(ckpt_path),
            "n_eval_puzzles": n,
            "n_chains": res.n_chains,
            "max_rounds": eval_max_rounds,
            "step_cfg": asdict(step_cfg),
            "run_args": {name: _arg_values[name] for name, _ in _RUN_PARAMS},
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
                "order": int(eval_orders_per_puzzle[i]),
                "correct": is_correct,
                "wrong": is_wrong,
                "timeout": is_timeout,
                "round_solved": rs,
                "n_resets": int(res.n_resets[i].item()),
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
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_train_puzzles: int | None = None,
    train_orders: str = "",
    eval_orders: str = "",
    translate_aug: bool = False,
    use_rope: bool = False,
    ckpt_name: str = "",
    ckpt_subdir: str = "",
    overwrite: bool = False,
    skip_if_done: bool = False,
):
    result = run.remote(
        steps=steps, batch_size=batch_size,
        n_eval_puzzles=n_eval_puzzles, seed=seed,
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
        use_ema=use_ema,
        ema_decay=ema_decay,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        eval_dropout_p=eval_dropout_p,
        n_train_puzzles=n_train_puzzles,
        train_orders=train_orders,
        eval_orders=eval_orders,
        translate_aug=translate_aug,
        use_rope=use_rope,
        ckpt_name=ckpt_name,
        ckpt_subdir=ckpt_subdir,
        overwrite=overwrite,
        skip_if_done=skip_if_done,
    )
    print(f"\nFinal: {result}", flush=True)
