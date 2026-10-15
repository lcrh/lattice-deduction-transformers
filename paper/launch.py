"""Launch, track and collect the runs of a paper manifest.

    uv run python -m paper.launch status  <manifest> [--offline]
    uv run python -m paper.launch launch  <manifest> [--dry-run] [--only GLOB] [--limit N]
    uv run python -m paper.launch collect <manifest> [--rows]
    uv run python -m paper.launch ledger  [--exp A1]
    uv run python -m paper.launch list

Every run is resolved locally (paper/manifest.py) to its checkpoint path and
eval summary path, so status is read off the checkpoint volume: a run is done
when its summary (or, for train-only runs, its checkpoint) exists. The launch
ledger (`ledger/` on the volume, written by the entrypoints) adds which runs
are in flight and their Modal app and call IDs.

`launch` submits only what is missing, one `modal run --detach -n <name>`
per run, without waiting. A run whose checkpoint exists is only evaluated;
eval-only runs wait for their training run. Launching refuses a dirty tree.
`collect` writes paper/results/<manifest>.csv (and with --rows the reported
per-puzzle rows) for the table and figure scripts.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable

from lattice_diffusion.provenance import (
    LedgerRecord, app_run_name, read_ledger, require_clean_sha,
)

from paper.manifest import (
    REPO, Run, eval_overrides, list_manifests, load_manifest, train_overrides, volume_path,
)
from paper.volume import ModalVolumeView, SnapshotVolumeView, VolumeView

RESULTS = REPO / "paper" / "results"

# Status values, in display order.
STATES = ("done", "running", "launched", "partial", "trained", "blocked", "failed", "not started")


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def modal_call_state(call_id: str) -> str:
    """running | finished | failed for a Modal function call. A finished
    training call that spawned an eval reports that eval's state."""
    import modal

    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return "running"
    except Exception as exc:
        if type(exc).__name__ == "TimeoutError":    # modal.exception.TimeoutError
            return "running"
        return "failed"
    if isinstance(result, dict) and result.get("eval_call_id"):
        return modal_call_state(result["eval_call_id"])
    return "finished"


def latest_ledger(records: list[LedgerRecord]) -> dict[str, LedgerRecord]:
    """Most recent ledger record per entry key."""
    out: dict[str, LedgerRecord] = {}
    for r in sorted(records, key=lambda r: r.ts):
        out[r.entry] = r
    return out


def run_state(run: Run, view: VolumeView, ledger: dict[str, LedgerRecord],
              call_state: Callable[[str], str] | None) -> str:
    """One of STATES. `call_state=None` means offline (no call lookups)."""
    if run.kind == "command":
        return "done" if run.result and (REPO / run.result).exists() else "not started"
    trained = view.exists(volume_path(run.checkpoint))
    if run.kind == "train":
        done = trained
    else:
        done = run.spec is not None and view.exists(volume_path(run.summary_file))
    if done:
        return "done"
    if run.kind == "eval" and run.spec is None:
        return "blocked"
    record = ledger.get(run.key)
    if record is not None:
        if call_state is None or record.call_id is None:
            return "launched"
        state = call_state(record.call_id)
        if state == "running":
            return "running"
        if state == "failed":
            return "failed"
    if run.spec is not None and view.listdir(volume_path(run.run_dir) + "/blocks"):
        return "partial"
    if run.kind == "eval" and not trained:
        return "blocked"
    if trained:
        return "trained"
    return "not started" if record is None else "failed"


def statuses(runs: list[Run], view: VolumeView, ledger: list[LedgerRecord],
             call_state: Callable[[str], str] | None) -> dict[str, str]:
    latest = latest_ledger(ledger)
    return {r.key: run_state(r, view, latest, call_state) for r in runs}


def load_ledger(view: VolumeView) -> list[LedgerRecord]:
    try:
        return read_ledger(view.listdir, view.read)
    except RuntimeError:        # offline snapshot: names only, no contents
        return []


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------

def _modal_run(name: str, script: str, args: list[str]) -> list[str]:
    return ["uv", "run", "modal", "run", "--detach", "-n", name, script, *args]


def train_command(run: Run) -> list[str]:
    args = []
    if run.config:
        args += ["--config", run.config]
    args += ["--set", shlex.join(train_overrides(run)), "--no-wait"]
    if run.kind == "train":
        args.append("--no-evaluate")
    return _modal_run(run.app_name, f"experiments/{run.benchmark}/run.py", args)


def eval_command(run: Run) -> list[str]:
    name = app_run_name(run.exp_id, f"{run.entry}-s{run.seed}", run.train_hash, run.eval_hash)
    args = ["--checkpoint", run.checkpoint, "--set", shlex.join(eval_overrides(run)),
            "--no-wait"]
    if run.indices_file:
        args += ["--indices-file", run.indices_file, "--split", run.split or "test"]
    return _modal_run(name, "experiments/evaluate.py", args)


def command_command(run: Run) -> list[str]:
    # Baselines write their result locally at the end, so no --detach.
    name = app_run_name(run.exp_id, run.entry, "baseline")
    return ["uv", "run", "modal", "run", "-n", name, *run.command]


def plan_launches(runs: list[Run], states: dict[str, str],
                  retry_failed: bool = False) -> list[tuple[Run, list[str] | None, str]]:
    """(run, command or None, reason) for every run, in manifest order.
    Runs that share a checkpoint are trained once."""
    out = []
    training: set[str] = set()
    for run in runs:
        state = states[run.key]
        if state in ("done", "running", "launched", "blocked"):
            out.append((run, None, state))
            continue
        if state == "failed" and not retry_failed:
            out.append((run, None, "failed (use --retry-failed)"))
            continue
        if run.kind == "command":
            out.append((run, command_command(run), "baseline eval"))
        elif state in ("trained", "partial") and run.kind != "train":
            out.append((run, eval_command(run), "evaluate existing checkpoint"))
        elif run.kind == "eval":
            out.append((run, None, "blocked: training not finished"))
        elif run.checkpoint in training:
            out.append((run, None, "waits for a run training the same checkpoint"))
        else:
            training.add(run.checkpoint)
            out.append((run, train_command(run), "train" if run.kind == "train"
                        else "train + evaluate"))
    return out


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

_DROP_OVERRIDES = ("provenance.", "train.seed", "eval.eval_seed")


def _entry_set(run: Run) -> dict:
    return {k: v for k, v in run.overrides.items() if not k.startswith(_DROP_OVERRIDES)}


def _safe(key: str) -> str:
    return key.replace("/", "__")


def run_record(run: Run, state: str) -> dict:
    rec = {
        "manifest": run.manifest, "exp_id": run.exp_id, "entry": run.entry, "key": run.key,
        "kind": run.kind, "label": run.label, "group": run.group, "x": run.x,
        "seed": run.seed, "eval_seed": run.eval_seed, "benchmark": run.benchmark,
        "config": run.config, "set": json.dumps(_entry_set(run), sort_keys=True),
        "checkpoint": run.checkpoint, "train_hash": run.train_hash,
        "eval_hash": run.eval_hash, "run_dir": run.run_dir, "status": state,
    }
    if run.exp is not None:
        m, t = run.exp.model, run.exp.train
        rec.update({
            "model_dim": m.dim, "model_layers": m.num_layers, "model_loops": m.n_loops,
            "model_heads": m.n_heads, "model_seq_len": m.seq_len, "model_ffn_mult": m.ffn_mult,
            "model_channels": m.n_channels, "model_cls": m.cls_token, "model_carry": m.carry,
            "train_steps": t.steps, "train_batch": t.batch_size,
            "solve_chains": run.exp.solve.n_chains, "solve_max_rounds": run.exp.solve.max_rounds,
        })
    return rec


def summary_fields(summary: dict) -> dict:
    lo, hi = summary["accuracy_cs"]
    calls, calls_c = summary["calls_all"], summary["calls_correct"]
    sound = summary.get("soundness") or {}
    return {
        "n": summary["n"], "stop_reason": summary["stop_reason"],
        "correct": summary["correct"], "wrong": summary["wrong"],
        "timeout": summary["timeout"], "accuracy": summary["accuracy"],
        "acc_lo": lo, "acc_hi": hi, "wrong_rate_upper": summary["wrong_rate_upper"],
        "failure_rate": summary["failure_rate"],
        "calls_mean": calls["mean"], "calls_p50": calls["p50"], "calls_p75": calls["p75"],
        "calls_p90": calls["p90"], "calls_p95": calls["p95"],
        "calls_correct_mean": calls_c["mean"],
        "unsound_rate": sound.get("unsound_rate"),
        "gpu_seconds": summary["gpu_seconds"], "startup_seconds": summary["startup_seconds"],
        "wall_seconds": summary["wall_seconds"], "n_evaluated": summary["n_evaluated"],
    }


def reported_rows(view: VolumeView, run: Run, n: int) -> list[dict]:
    """The rows behind a summary: positions 0..n-1 of the run's index list."""
    blocks_dir = volume_path(run.run_dir) + "/blocks"
    rows = []
    for name in sorted(view.listdir(blocks_dir)):
        if not name.endswith(".jsonl"):
            continue
        for line in view.read(f"{blocks_dir}/{name}").decode().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["pos"] < n:
                    rows.append(row)
    return sorted(rows, key=lambda r: r["pos"])


def collect(manifest: str, runs: list[Run], view: VolumeView, states: dict[str, str],
            results_dir: Path = RESULTS, with_rows: bool = False) -> Path:
    records = []
    for run in runs:
        rec = run_record(run, states[run.key])
        if run.kind == "command":
            if rec["status"] == "done":
                rec["result_path"] = run.result
        elif run.kind == "train":
            pass
        elif rec["status"] == "done":
            summary = json.loads(view.read(volume_path(run.summary_file)))
            rec.update(summary_fields(summary))
            if with_rows:
                path = results_dir / "rows" / manifest / f"{_safe(run.key)}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = reported_rows(view, run, summary["n"])
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                rec["rows_path"] = str(path.relative_to(REPO)) if path.is_relative_to(REPO) \
                    else str(path)
        if run.checkpoint and view.exists(volume_path(run.train_record)):
            rec["train_seconds"] = json.loads(view.read(volume_path(run.train_record)))[
                "train_seconds"]
        records.append(rec)

    out = results_dir / f"{manifest}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for rec in records for k in rec))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _view(offline: bool) -> VolumeView:
    return SnapshotVolumeView() if offline else ModalVolumeView()


def _filter(runs: list[Run], only: str | None) -> list[Run]:
    if not only:
        return runs
    return [r for r in runs if fnmatch.fnmatch(r.entry, only) or fnmatch.fnmatch(r.key, only)]


def _print_status(runs: list[Run], states: dict[str, str], ledger: list[LedgerRecord]) -> None:
    latest = latest_ledger(ledger)
    width = max((len(r.key) for r in runs), default=10)
    for run in runs:
        rec = latest.get(run.key)
        app = f"  app={rec.app_id}" if rec and states[run.key] in ("running", "launched") else ""
        print(f"  {run.key:<{width}}  {states[run.key]:<11} {run.kind:<10}{app}")
    counts = {s: sum(v == s for v in states.values()) for s in STATES}
    print("  " + "  ".join(f"{s}={c}" for s, c in counts.items() if c))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paper.launch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "launch", "collect"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.add_argument("--only", help="glob over entry names or keys")
        p.add_argument("--offline", action="store_true",
                       help="use the last saved volume listing; no network")
    sub.choices["launch"].add_argument("--dry-run", action="store_true")
    sub.choices["launch"].add_argument("--limit", type=int, default=None)
    sub.choices["launch"].add_argument("--retry-failed", action="store_true")
    sub.choices["collect"].add_argument("--rows", action="store_true",
                                        help="also download the reported per-puzzle rows")
    p = sub.add_parser("ledger")
    p.add_argument("--exp")
    p.add_argument("--offline", action="store_true")
    sub.add_parser("list")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(list_manifests()))
        return
    offline = getattr(args, "offline", False) or getattr(args, "dry_run", False)
    view = _view(offline)
    ledger = load_ledger(view)
    if args.cmd == "ledger":
        for r in ledger:
            if args.exp and r.exp_id != args.exp:
                continue
            print(json.dumps(r.__dict__))
        return

    _, runs = load_manifest(args.manifest)
    runs = _filter(runs, args.only)
    states = statuses(runs, view, ledger, None if offline else modal_call_state)
    if isinstance(view, ModalVolumeView):
        view.save_snapshot()

    if args.cmd == "status":
        _print_status(runs, states, ledger)
    elif args.cmd == "collect":
        print(f"wrote {collect(args.manifest, runs, view, states, with_rows=args.rows)}")
    elif args.cmd == "launch":
        plan = plan_launches(runs, states, args.retry_failed)
        todo = [(r, c) for r, c, _ in plan if c is not None][:args.limit]
        for run, cmd, reason in plan:
            print(f"  {run.key:<40} {reason}")
        if args.dry_run:
            for _, cmd in todo:
                print("  $ " + shlex.join(cmd))
            return
        sha = require_clean_sha(REPO)
        for run, cmd in todo:
            print(f"\n$ {shlex.join(cmd)}", flush=True)
            result = subprocess.run(cmd, cwd=REPO)
            if result.returncode != 0:
                print(f"launch of {run.key} failed (exit {result.returncode})", file=sys.stderr)
            elif run.kind == "command":
                # Baseline scripts do not write the ledger themselves.
                _ledger_command(run, cmd, sha)


def _ledger_command(run: Run, cmd: list[str], sha: str) -> None:
    import modal

    from lattice_diffusion.provenance import announce, write_ledger
    from paper.volume import VOLUME_NAME

    record = LedgerRecord(kind="command", exp_id=run.exp_id, entry=run.key,
                          app_name=cmd[cmd.index("-n") + 1], app_id=None, app_url=None,
                          call_id=None, code_sha=sha, checkpoint="", train_hash="",
                          eval_run_dir=run.result)
    announce(record, write_ledger(modal.Volume.from_name(VOLUME_NAME), record))


if __name__ == "__main__":
    main()
