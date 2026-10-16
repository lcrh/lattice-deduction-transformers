"""Launch, track and collect the runs of a paper manifest.

    uv run python -m paper.launch status  <manifest> [--offline] [--only GLOB]
    uv run python -m paper.launch launch  <manifest> [--dry-run] [--only GLOB] [--limit N]
                                          [--retry-failed]
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
import dataclasses
import fnmatch
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, get_args

from lattice_diffusion.provenance import (
    LedgerRecord, announce, app_run_name, read_ledger, require_clean_sha, write_ledger,
)

from paper.manifest import (
    REPO, Run, eval_overrides, list_manifests, load_manifest, train_overrides, volume_path,
)
from paper.results import RESULTS
from paper.volume import VOLUME_NAME, ModalVolumeView, SnapshotVolumeView, VolumeView

State = Literal["done", "running", "launched", "partial", "trained", "blocked", "failed",
                "not started"]
STATES = get_args(State)      # display order
CallState = Literal["running", "finished", "failed"]


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def modal_call_state(call_id: str) -> CallState:
    """State of a Modal function call; a finished training call reports its spawned eval's."""
    import modal

    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        return "running"
    except Exception:
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
              call_state: Callable[[str], CallState] | None) -> State:
    """A run's state; `call_state=None` means offline (no call lookups)."""
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
             call_state: Callable[[str], CallState] | None) -> dict[str, State]:
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


@dataclass
class Launch:
    """What `launch` does for one run: a command, or None with the reason it is skipped."""

    run: Run
    cmd: list[str] | None
    reason: str
    app_name: str | None = None


def train_command(run: Run) -> list[str]:
    args = []
    if run.config:
        args += ["--config", run.config]
    args += ["--set", shlex.join(train_overrides(run)), "--no-wait"]
    if run.kind == "train":
        args.append("--no-evaluate")
    return _modal_run(run.app_name(), f"experiments/{run.benchmark}/run.py", args)


def eval_command(run: Run) -> list[str]:
    args = ["--checkpoint", run.checkpoint, "--set", shlex.join(eval_overrides(run)),
            "--no-wait"]
    if run.indices_file:
        args += ["--indices-file", run.indices_file, "--split", run.split or "test"]
    return _modal_run(run.app_name(with_eval=True), "experiments/evaluate.py", args)


def baseline_name(run: Run) -> str:
    return app_run_name(run.exp_id, run.entry, "baseline")


def baseline_command(run: Run) -> list[str]:
    # Baselines write their result locally at the end, so no --detach.
    return ["uv", "run", "modal", "run", "-n", baseline_name(run), *run.command]


def plan_launches(runs: list[Run], states: dict[str, State],
                  retry_failed: bool = False) -> list[Launch]:
    """One Launch per run, in manifest order; runs that share a checkpoint train once."""
    out = []
    training: set[str] = set()
    for run in runs:
        state = states[run.key]
        if state in ("done", "running", "launched", "blocked"):
            out.append(Launch(run, None, state))
        elif state == "failed" and not retry_failed:
            out.append(Launch(run, None, "failed (use --retry-failed)"))
        elif run.kind == "command":
            out.append(Launch(run, baseline_command(run), "baseline eval", baseline_name(run)))
        elif state in ("trained", "partial") and run.kind != "train":
            out.append(Launch(run, eval_command(run), "evaluate existing checkpoint",
                              run.app_name(with_eval=True)))
        elif run.kind == "eval":
            out.append(Launch(run, None, "blocked: training not finished"))
        elif run.checkpoint in training:
            out.append(Launch(run, None, "waits for a run training the same checkpoint"))
        else:
            training.add(run.checkpoint)
            out.append(Launch(run, train_command(run),
                              "train" if run.kind == "train" else "train + evaluate",
                              run.app_name()))
    return out


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

_DROP_OVERRIDES = ("provenance.", "train.seed", "eval.eval_seed")


def _entry_set(run: Run) -> dict:
    return {k: v for k, v in run.overrides.items() if not k.startswith(_DROP_OVERRIDES)}


def _safe(key: str) -> str:
    return key.replace("/", "__")


def run_record(run: Run, state: State) -> dict:
    """The CSV columns describing a run (identity, settings, state)."""
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
    """The CSV columns taken from an eval summary."""
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
        "model_calls_total": summary["model_calls_total"],
        "mean_rounds_correct": summary["mean_rounds_correct"],
        "mean_resets": summary["mean_resets"],
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


def _display_path(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def collect(manifest: str, runs: list[Run], view: VolumeView, states: dict[str, State],
            results_dir: Path = RESULTS, with_rows: bool = False) -> Path:
    """Write paper/results/<manifest>.csv (and with `with_rows`, the reported rows)."""
    records = []
    for run in runs:
        rec = run_record(run, states[run.key])
        if rec["status"] == "done" and run.kind == "command":
            rec["result_path"] = run.result
        elif rec["status"] == "done" and run.kind in ("eval", "train+eval"):
            summary = json.loads(view.read(volume_path(run.summary_file)))
            rec.update(summary_fields(summary))
            if with_rows:
                path = results_dir / "rows" / manifest / f"{_safe(run.key)}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = reported_rows(view, run, summary["n"])
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                rec["rows_path"] = _display_path(path)
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
    sub.choices["launch"].add_argument("--dry-run", action="store_true",
                                       help="print the plan and commands; implies --offline")
    sub.choices["launch"].add_argument("--limit", type=int, default=None,
                                       help="launch at most this many runs")
    sub.choices["launch"].add_argument("--retry-failed", action="store_true",
                                       help="relaunch runs whose last launch failed")
    sub.choices["collect"].add_argument("--rows", action="store_true",
                                        help="also download the reported per-puzzle rows")
    p = sub.add_parser("ledger")
    p.add_argument("--exp", help="only this experiment ID")
    p.add_argument("--offline", action="store_true", help="use the last saved volume listing")
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
            print(json.dumps(dataclasses.asdict(r)))
        return

    runs = load_manifest(args.manifest)
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
        todo = [launch for launch in plan if launch.cmd is not None][:args.limit]
        for launch in plan:
            print(f"  {launch.run.key:<40} {launch.reason}")
        if args.dry_run:
            for launch in todo:
                print("  $ " + shlex.join(launch.cmd))
            return
        sha = require_clean_sha(REPO)
        for launch in todo:
            print(f"\n$ {shlex.join(launch.cmd)}", flush=True)
            result = subprocess.run(launch.cmd, cwd=REPO)
            if result.returncode != 0:
                print(f"launch of {launch.run.key} failed (exit {result.returncode})",
                      file=sys.stderr)
            elif launch.run.kind == "command":
                _ledger_baseline(launch, sha)     # baseline scripts do not write the ledger


def _ledger_baseline(launch: Launch, sha: str) -> None:
    import modal

    run = launch.run
    record = LedgerRecord(kind="command", exp_id=run.exp_id, entry=run.key,
                          app_name=launch.app_name, app_id=None, app_url=None,
                          call_id=None, code_sha=sha, checkpoint="", train_hash="",
                          eval_run_dir=run.result)
    announce(record, write_ledger(modal.Volume.from_name(VOLUME_NAME), record))


if __name__ == "__main__":
    main()
