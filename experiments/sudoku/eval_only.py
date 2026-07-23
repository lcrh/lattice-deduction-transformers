"""Eval an existing sudoku checkpoint, fanned out across N Modal workers.

Every sudoku-extreme puzzle is SAT, and the eval config is zero-hint +
no-augment, so a sample is *exactly* the cached `(x, y)` tensors (the
`zero_hints` branch of `_make_sample` returns the puzzle verbatim and the
no-flag `_augment` is the identity). We therefore skip the streaming sampler
and load the cache directly.

Each worker (one B200 each) loads the cached pool, takes a strided slice
(`worker_idx::n_workers`) so every worker sees an equally-mixed difficulty
distribution, solves it, and returns per-puzzle rows + diag aggregates that
the driver merges. `--workers 1` is the single-container path.

Usage:
    # Whole test split across 10 B200 workers
    uv run modal run --detach experiments/sudoku/eval_only.py \
      --checkpoint /checkpoints/sudoku/seed0_4000s_bs512_aug1_<ts>.pt \
      --n-eval -1 --workers 10

    # Canonical 1,000-puzzle subset on one worker
    uv run modal run --detach experiments/sudoku/eval_only.py \
      --checkpoint /checkpoints/sudoku/seed0_4000s_bs512_aug1_<ts>.pt \
      --n-eval 1000
"""

import json
import time
from dataclasses import dataclass

import modal
import numpy as np
import torch
from torch import nn

from lattice_diffusion.data.sudoku_extreme import _download_and_cache
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


@dataclass
class _WorkerArgs:
    """Self-contained eval-config bundle handed to each worker. All fields are
    scalars / strings so it ships cleanly across Modal RPC. Each worker derives
    its own puzzle slice from (worker_idx, n_workers), so the driver does no
    GPU work and never has to pre-count the pool."""
    checkpoint: str
    n_eval: int
    worker_idx: int
    n_workers: int
    threshold: float
    temp_decide: float
    cls_threshold: float
    n_chains: int
    batch_size: int
    max_rounds: int
    augment: bool
    estimate_sequential: bool
    seq_drain_max_rounds: int
    dropout_p: float
    log_per_round_fill: bool
    split: str
    subset: list[int]   # explicit puzzle-idx pool to eval (empty = whole split)


def _load_pool(split: str, n_eval: int):
    """Load the cached eval pool as float tensors `(x, y)` of shape [N, 81, 9].
    `x` has givens one-hot and blanks all-ones; `y` is the one-hot solution.

    `n_eval <= 0` uses the whole split. `0 < n_eval < N` reproduces
    SudokuExtremeDataset's seed=200 subset selection exactly, so it matches the
    canonical subset that run.py evals."""
    cache_path = _download_and_cache(DATA_MOUNT, split)
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    x, y = data["x"], data["y"]  # uint8 [N, 81, 9]
    n_total = x.shape[0]
    if 0 < n_eval < n_total:
        idx = np.sort(np.random.default_rng(200).choice(n_total, size=n_eval, replace=False))
        x, y = x[idx], y[idx]
    return x.float(), y.float()


def _summarize_solve_result(
    res, indices: list[int], n_chains: int, max_rounds: int,
    estimate_sequential: bool, log_per_round_fill: bool,
) -> dict:
    """Convert a SolveResult over `indices` puzzles into a dict that serializes
    cleanly across Modal RPC: per-puzzle rows (tagged with the original
    puzzle_idx) + aggregate diag totals the driver sums."""
    # Encode each puzzle's final predicted grid as an 81-char digit string
    # ('0' = unresolved/ambiguous cell), so downstream analysis (e.g. majority
    # voting across seeds, inspecting wrong answers) has the actual grid the
    # model accepted, not just the correct/wrong/timeout flag. The givens and
    # ground truth are recoverable from the cache via puzzle_idx, so we only
    # store the model's own output here.
    sol_single = (res.solution.sum(dim=-1) == 1)              # [M, S]
    sol_digits = ((res.solution.argmax(dim=-1) + 1) * sol_single.long()).cpu().tolist()
    solution_strs = ["".join(str(d) for d in cells) for cells in sol_digits]

    rows: list[dict] = []
    for j, orig_i in enumerate(indices):
        is_correct = bool(res.correct[j].item())
        is_wrong = bool(res.wrong[j].item())
        is_timeout = bool(res.timeouts[j].item())
        rs = int(res.round_solved[j].item())
        fwd = (rs + 1) * n_chains if is_correct else max_rounds * n_chains
        row = {
            "kind": "puzzle",
            "puzzle_idx": orig_i,
            "correct": is_correct,
            "wrong": is_wrong,
            "timeout": is_timeout,
            "round_solved": rs,
            "n_resets": int(res.n_resets[j].item()),
            "forwards_unbatched": fwd,
            "solution": solution_strs[j],
        }
        if estimate_sequential:
            seq_v = int(res.forwards_seq[j].item())
            w_idx = int(res.seq_winning_idx[j].item())
            done = int(res.seq_attempts_done[j].item())
            row["forwards_seq"] = seq_v
            row["seq_winning_idx"] = w_idx
            row["seq_attempts_done"] = done
            row["seq_avg_attempt_len"] = (
                (seq_v / max(w_idx + 1, 1)) if w_idx >= 0 else None
            )
        if log_per_round_fill and is_correct:
            row["n_givens"] = res.n_givens[j]
            row["deduction_fills_per_round"] = res.deduction_fills_per_round[j]
            row["decision_fills_per_round"] = res.decision_fills_per_round[j]
            row["deduction_bitflips_per_round"] = res.deduction_bitflips_per_round[j]
            row["decision_bitflips_per_round"] = res.decision_bitflips_per_round[j]
        rows.append(row)

    return {
        "rows": rows,
        "diag": {
            "model_calls": int(res.model_calls),
            "total_deduced": int(res.diag_total_deduced),
            "total_unsound_deductions": int(res.diag_total_unsound_deductions),
            "conflict_tp": int(res.diag_conflict_tp),
            "conflict_fp": int(res.diag_conflict_fp),
            "conflict_fn": int(res.diag_conflict_fn),
            "conflict_tn": int(res.diag_conflict_tn),
            "active_chain_rounds": int(res.diag_active_chain_rounds),
        },
    }


@app.function(
    image=image, gpu="B200", timeout=21600,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def eval_chunk(args: _WorkerArgs) -> dict:
    """Load the pool, solve this worker's strided slice, return its rows."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    print(f"[worker {args.worker_idx}] loading {args.checkpoint}", flush=True)
    ckpt = load_checkpoint(args.checkpoint)
    cfg = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if args.dropout_p > 0.0:
        n_drop = 0
        n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = args.dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = args.dropout_p
                n_mha += 1
        model.train()
        print(f"[worker {args.worker_idx}] dropout-noise: overrode {n_drop} Dropout "
              f"+ {n_mha} MHA-attn-dropout to p={args.dropout_p}, model in train()",
              flush=True)
    if device.type == "cuda":
        print(f"[worker {args.worker_idx}] torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)

    # An explicit subset addresses absolute cache indices, so it only makes
    # sense against the whole split — force a full-split load when given.
    x_full, y_full = _load_pool(args.split, -1 if args.subset else args.n_eval)
    n_total = x_full.shape[0]
    # Candidate pool: the explicit subset (e.g. a residual hard set) if given,
    # else the whole split. Each worker takes a strided slice of that pool.
    pool = args.subset if args.subset else list(range(n_total))
    indices = pool[args.worker_idx::args.n_workers]
    if not indices:
        print(f"[worker {args.worker_idx}] no puzzles in slice", flush=True)
        return {"rows": [], "diag": {k: 0 for k in (
            "model_calls", "total_deduced", "total_unsound_deductions",
            "conflict_tp", "conflict_fp", "conflict_fn", "conflict_tn",
            "active_chain_rounds")}, "wall_seconds": 0.0, "n_total": n_total}

    idx_t = torch.tensor(indices, dtype=torch.long)
    x = x_full.index_select(0, idx_t).to(device)
    y = y_full.index_select(0, idx_t).to(device)
    given_mask = (x.sum(dim=-1) == 1)
    print(f"[worker {args.worker_idx}] pool={n_total}, this worker handles "
          f"{len(indices)} puzzles ({indices[0]}..{indices[-1]} stride {args.n_workers})",
          flush=True)

    step_cfg = StepConfig(
        threshold=args.threshold,
        temp_decide=args.temp_decide,
        cls_threshold=args.cls_threshold,
        augment=args.augment,
    )
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=args.max_rounds,
        n_chains=args.n_chains, batch_size=args.batch_size,
        estimate_sequential=args.estimate_sequential,
        seq_drain_max_rounds=args.seq_drain_max_rounds,
        log_per_round_fill=args.log_per_round_fill,
    )

    t0 = time.time()
    res = solve(model, x, y, given_mask, solve_cfg)
    elapsed = time.time() - t0
    print(f"[worker {args.worker_idx}] solve done in {elapsed:.0f}s", flush=True)

    summary = _summarize_solve_result(
        res, indices, args.n_chains, args.max_rounds,
        args.estimate_sequential, args.log_per_round_fill,
    )
    summary["wall_seconds"] = elapsed
    summary["n_total"] = n_total
    return summary


@app.function(
    image=image, gpu=None, timeout=120,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def _write_eval_outputs(eval_path: str, eval_jsonl_path: str,
                        summary: dict, rows: list[dict]) -> None:
    """Write the merged summary JSON + per-puzzle JSONL onto the checkpoint
    volume (not mounted on the local driver)."""
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_jsonl_path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **summary}) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    checkpoint_volume.commit()


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
    workers: int = 1,
    puzzle_indices: str = "",
):
    # Optional explicit puzzle-idx pool (comma-separated) — e.g. a residual
    # hard set — so we can cheaply eval new checkpoints on just those puzzles
    # instead of the whole split.
    subset = [int(s) for s in puzzle_indices.split(",") if s.strip()] if puzzle_indices else []
    workers = max(1, min(workers, len(subset) if subset else workers))
    scope = f"{len(subset)} puzzles (subset)" if subset else f"whole split ({split})"
    print(f"[driver] fanning out to {workers} workers (n_eval={n_eval}, {scope})",
          flush=True)

    worker_args = [
        _WorkerArgs(
            checkpoint=checkpoint, n_eval=n_eval,
            worker_idx=k, n_workers=workers,
            threshold=threshold, temp_decide=temp_decide,
            cls_threshold=cls_threshold,
            n_chains=n_chains, batch_size=batch_size, max_rounds=max_rounds,
            augment=augment,
            estimate_sequential=estimate_sequential,
            seq_drain_max_rounds=seq_drain_max_rounds,
            dropout_p=dropout_p, log_per_round_fill=log_per_round_fill,
            split=split, subset=subset,
        )
        for k in range(workers)
    ]

    t0 = time.time()
    # `.map(args)` fans out each item to a fresh container, in parallel.
    parts = list(eval_chunk.map(worker_args))
    elapsed = time.time() - t0
    print(f"[driver] all workers done in {elapsed:.0f}s", flush=True)

    # Merge per-puzzle rows by puzzle_idx, sum diag aggregates.
    rows_by_idx: dict[int, dict] = {}
    accum = {
        "model_calls": 0,
        "total_deduced": 0,
        "total_unsound_deductions": 0,
        "conflict_tp": 0,
        "conflict_fp": 0,
        "conflict_fn": 0,
        "conflict_tn": 0,
        "active_chain_rounds": 0,
    }
    for part in parts:
        for row in part["rows"]:
            rows_by_idx[row["puzzle_idx"]] = row
        for k in accum:
            accum[k] += part["diag"][k]

    rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
    n = len(rows)

    n_correct = sum(1 for r in rows if r["correct"])
    n_wrong = sum(1 for r in rows if r["wrong"])
    n_timeout = sum(1 for r in rows if r["timeout"])
    avg_calls = accum["model_calls"] / max(n_correct, 1)
    den = max(accum["total_deduced"], 1)
    unsound_rate = accum["total_unsound_deductions"] / den
    cls_p = accum["conflict_tp"] / max(accum["conflict_tp"] + accum["conflict_fp"], 1)
    cls_r = accum["conflict_tp"] / max(accum["conflict_tp"] + accum["conflict_fn"], 1)

    print(f"\n{'='*60}\nRESULT ({n} puzzles, {len(parts)} workers)\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}", flush=True)
    print(f"  total_calls={accum['model_calls']}  avg/correct={avg_calls:.1f}", flush=True)
    print(f"  Deduction soundness: {accum['total_unsound_deductions']} unsound / "
          f"{accum['total_deduced']} deduced  (rate={unsound_rate:.4%})", flush=True)
    print(f"  Conflict head P={cls_p:.3f} R={cls_r:.3f} "
          f"[tp={accum['conflict_tp']} fp={accum['conflict_fp']} "
          f"fn={accum['conflict_fn']} tn={accum['conflict_tn']}] "
          f"over {accum['active_chain_rounds']} active chain-rounds", flush=True)
    print(f"  driver wall: {elapsed:.0f}s "
          f"(slowest worker: {max(p['wall_seconds'] for p in parts):.0f}s)",
          flush=True)

    out = {
        "checkpoint": checkpoint,
        "n_eval": n,
        "solver": "hybrid_per_chunk_parallel",
        "n_workers": len(parts),
        "wall_seconds_driver": elapsed,
        "wall_seconds_max_worker": max(p["wall_seconds"] for p in parts),
        "solver_config": {
            "threshold": threshold, "temp_decide": temp_decide,
            "cls_threshold": cls_threshold,
            "n_chains": n_chains, "batch_size": batch_size,
            "max_rounds": max_rounds,
            "augment": augment, "dropout_p": dropout_p,
            "split": split,
        },
        "summary": {
            "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
            "total_calls": accum["model_calls"],
            "avg_calls_per_correct": avg_calls,
        },
        "diag": {
            "total_deduced": accum["total_deduced"],
            "total_unsound_deductions": accum["total_unsound_deductions"],
            "unsound_rate": unsound_rate,
            "conflict_tp": accum["conflict_tp"],
            "conflict_fp": accum["conflict_fp"],
            "conflict_fn": accum["conflict_fn"],
            "conflict_tn": accum["conflict_tn"],
            "conflict_precision": cls_p,
            "conflict_recall": cls_r,
            "active_chain_rounds": accum["active_chain_rounds"],
        },
    }
    eval_path = checkpoint.replace(".pt", out_suffix)
    eval_jsonl_path = checkpoint.replace(".pt", out_suffix.replace(".json", ".jsonl"))

    # Write output via a tiny modal helper since the volume isn't mounted
    # locally.
    _write_eval_outputs.remote(eval_path, eval_jsonl_path, out, rows)
    print(f"Wrote {eval_path}", flush=True)
    print(f"Wrote {eval_jsonl_path}", flush=True)


@app.local_entrypoint()
def sweep(
    checkpoints: str,            # comma-separated checkpoint paths
    puzzle_indices: str,         # comma-separated puzzle-idx (the hard set)
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    dropout_p: float = 0.05,
    split: str = "test",
):
    """Eval many checkpoints on the SAME explicit puzzle subset (e.g. a hard
    set), one container per checkpoint, and report each checkpoint's solved
    count plus the combined residue (puzzles no checkpoint solves)."""
    ckpts = [c.strip() for c in checkpoints.split(",") if c.strip()]
    subset = [int(s) for s in puzzle_indices.split(",") if s.strip()]
    print(f"[sweep] {len(ckpts)} checkpoints × {len(subset)} puzzles", flush=True)

    args_list = [
        _WorkerArgs(
            checkpoint=c, n_eval=-1, worker_idx=0, n_workers=1,
            threshold=threshold, temp_decide=temp_decide, cls_threshold=cls_threshold,
            n_chains=n_chains, batch_size=batch_size, max_rounds=max_rounds,
            augment=augment, estimate_sequential=False, seq_drain_max_rounds=200,
            dropout_p=dropout_p, log_per_round_fill=False,
            split=split, subset=subset,
        )
        for c in ckpts
    ]
    # one container per checkpoint
    parts = list(eval_chunk.map(args_list))

    union = set()
    # per-puzzle stats across checkpoints: # solvers + rounds-to-solve list
    per_puzzle = {i: {"solved": 0, "rounds": []} for i in subset}
    print("\n[sweep] per-checkpoint solved (of "
          f"{len(subset)}):", flush=True)
    for c, p in zip(ckpts, parts):
        solved = {row["puzzle_idx"] for row in p["rows"] if row["correct"]}
        union |= solved
        for row in p["rows"]:
            if row["correct"]:
                per_puzzle[row["puzzle_idx"]]["solved"] += 1
                per_puzzle[row["puzzle_idx"]]["rounds"].append(row["round_solved"])
        name = c.rsplit("/", 1)[-1]
        print(f"  {name}: {len(solved)}", flush=True)
    residue = sorted(set(subset) - union)
    print(f"\n[sweep] union solved: {len(union)}/{len(subset)}  "
          f"residue (no ckpt solves): {len(residue)}", flush=True)
    print(f"[sweep] residue idx: {residue}", flush=True)

    # Per-puzzle hardness ranking: fewest solvers = hardest; tie-break on
    # higher mean rounds-to-solve among the checkpoints that did solve it.
    print(f"\n[sweep] per-puzzle hardness ({len(ckpts)} checkpoints):", flush=True)
    def _mean(xs): return sum(xs) / len(xs) if xs else None
    ranked = sorted(subset, key=lambda i: (per_puzzle[i]["solved"],
                                           -(_mean(per_puzzle[i]["rounds"]) or 0)))
    for i in ranked:
        st = per_puzzle[i]
        mr = _mean(st["rounds"])
        mr_s = f"{mr:.0f}" if mr is not None else "—"
        mx = max(st["rounds"]) if st["rounds"] else "—"
        print(f"  idx={i}: solved by {st['solved']}/{len(ckpts)}  "
              f"mean_rounds={mr_s}  max_rounds={mx}", flush=True)
