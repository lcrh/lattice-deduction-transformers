"""Steady-state training throughput samples on a single B200, for the E8 cost
table's GPU-hours normalization.

WHY. The repro study already measured TRM/HRM Maze-Hard steady-state s/step on
B200 (repro/README.md: TRM ~1.46, HRM ~0.61 s/step at batch 768). The one
missing same-silicon cell is TRM on SUDOKU (reported by the TRM authors as
4x L40S x 36h, a different GPU generation). Rather than retrain to convergence,
we sample ~100 steps on B200 and read off steady-state s/step; GPU-hours then =
steps_to_convergence * s_per_step. This entrypoint produces that number (and can
sample any other missing model x task cell the same way).

HOW IT REUSES THE REPRO HARNESS. It reuses repro/trm_eval/modal_train.py's
Modal image verbatim (imported), the same TRM commit, the same dataset-build
step, and the same `scripts/train.py` hydra invocation. The ONLY addition is a
`--step-cap` (default a large 100 for the measurement job) that:
  * appends `eval_interval=<huge>` so no in-train eval pollutes the timing,
  * streams train.py's stdout, parses per-step wall-clock from consecutive step
    log lines' timestamps, EXCLUDES step 1 (compile/JIT warmup), averages the
    steady-state middle, and TERMINATES the subprocess once step-cap steps have
    elapsed.
This wrapper does NOT modify the repro harness (which stays byte-identical and
default-full-length). It is a measurement-only side entrypoint.

LDT / sudoku (this repo's model) already records steady-state s/step natively:
experiments/sudoku/train.py writes `train_post_compile_secs` (compile excluded)
into the checkpoint, so LDT's s/step = train_post_compile_secs / (steps-1). No
separate harness needed for LDT — read it from any e1/e2/e3 eval.json's
train_wallclock block. This file covers the OTHER models.

WHAT TO LAUNCH (not run here — no Modal access in-agent):

    # TRM on SUDOKU — the missing same-silicon cell (arch=trm, sudoku config):
    uv run modal run followups/cost_accounting/measure_throughput.py::run \
        --model trm --task sudoku --step-cap 100

    # (re-)measure TRM or HRM on maze if you want to refresh the repro number:
    uv run modal run followups/cost_accounting/measure_throughput.py::run \
        --model trm --task maze --step-cap 100

The entrypoint prints a JSON blob: {model, task, steady_s_per_step,
n_steps_timed, batch}. Paste steady_s_per_step into cost_table.py's
MEASURED_THROUGHPUT map (source="measured").
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

# Reuse the repro TRM harness's Modal image + constants VERBATIM (no fork).
# `repro/` is NOT an installed package (no __init__.py; launched by path via
# `modal run repro/...`), so we load its module by file path rather than
# importing `repro.trm_eval.modal_train`. This picks up the SAME image object,
# TRM commit, and TRM_REMOTE that the repro training harness uses.
_REPRO_TRM = (Path(__file__).resolve().parents[2]
              / "repro" / "trm_eval" / "modal_train.py")
_spec = importlib.util.spec_from_file_location("_repro_trm_modal_train", _REPRO_TRM)
_repro_trm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repro_trm)
TRM_REMOTE = _repro_trm.TRM_REMOTE
trm_image = _repro_trm.image

app = modal.App("cost-measure-throughput")

# TRM sudoku recipe: same knobs as the maze recipe (repro TRAIN_ARGS) but
# pointed at the sudoku dataset build. arch=trm, batch 768 to match the maze
# measurement so per-step numbers are directly comparable. eval disabled and
# epochs tiny — we only need steady-state per-step wall-clock, not convergence.
_TRM_SUDOKU_ARGS = [
    "arch=trm",
    "data_paths=[data/sudoku-extreme-1k-aug-1000]",
    "evaluators=[]",
    "epochs=200", "eval_interval=100000",   # eval effectively off for the sample
    "lr=2e-4", "puzzle_emb_lr=1e-4",
    "weight_decay=1.0", "puzzle_emb_weight_decay=1.0",
    "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=4",
    "global_batch_size=768", "lr_warmup_steps=100",
]
_TRM_MAZE_ARGS = [
    "arch=trm",
    "data_paths=[data/maze-30x30-hard-1k]",
    "evaluators=[]",
    "epochs=200", "eval_interval=100000",
    "lr=2e-4", "puzzle_emb_lr=1e-4",
    "weight_decay=1.0", "puzzle_emb_weight_decay=1.0",
    "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=4",
    "global_batch_size=768", "lr_warmup_steps=100",
]

# TRM's train.py logs a "step" scalar to stdout each optimization step. We match
# any line carrying an integer step counter and timestamp its arrival locally.
_STEP_RE = re.compile(r"\bstep[\"'=: ]+(\d+)")


@app.function(image=trm_image, gpu="B200", timeout=1800)
def measure(model: str = "trm", task: str = "sudoku", step_cap: int = 100,
            batch: int = 768) -> dict:
    """Sample `step_cap` training steps of `model` on `task`, return
    steady-state s/step (step 1 excluded as compile/JIT warmup)."""
    if model != "trm":
        raise ValueError(
            f"model={model!r}: only TRM is wired here (HRM/LDT are already "
            f"measured — see module docstring). Add an args block to extend."
        )
    if task == "sudoku":
        base_args = list(_TRM_SUDOKU_ARGS)
        build_cmd = [sys.executable, "-m", "trm.data.build_sudoku_dataset",
                     "--output-dir", "data/sudoku-extreme-1k-aug-1000",
                     "--subsample-size", "1000", "--num-aug", "1000"]
    elif task == "maze":
        base_args = list(_TRM_MAZE_ARGS)
        build_cmd = [sys.executable, "-m", "trm.data.build_maze_dataset",
                     "--output-dir", "data/maze-30x30-hard-1k"]
    else:
        raise ValueError(f"unknown task {task!r} (expected 'sudoku' or 'maze')")

    # Allow the batch to be overridden without touching the recipe list.
    args = [a for a in base_args if not a.startswith("global_batch_size=")]
    args.append(f"global_batch_size={batch}")

    print(f"[measure] building {task} dataset for TRM …", flush=True)
    subprocess.run(build_cmd, cwd=TRM_REMOTE, check=True)

    cmd = [sys.executable, "scripts/train.py", *args,
           "+run_name=cost-measure", "+project_name=cost-measure"]
    env = {**os.environ, "WANDB_MODE": "disabled"}
    print(f"[measure] launching (step_cap={step_cap}):\n  {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, cwd=TRM_REMOTE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    step_times: list[tuple[int, float]] = []
    last_step = -1
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = _STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step == last_step:
                continue
            last_step = step
            step_times.append((step, time.perf_counter()))
            if len(step_times) >= step_cap + 2:  # +2: drop step-1 boundary + margin
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    if len(step_times) < 4:
        raise RuntimeError(
            f"only captured {len(step_times)} step log lines — could not "
            f"measure steady state (check the step-log regex vs train.py output)."
        )

    # Deltas between consecutive captured steps. Drop the FIRST delta (spans the
    # step-1 compile/JIT warmup) and take the median of the rest for robustness.
    deltas = []
    for (s0, t0), (s1, t1) in zip(step_times, step_times[1:]):
        ds = s1 - s0
        if ds <= 0:
            continue
        deltas.append((t1 - t0) / ds)
    steady = deltas[1:] if len(deltas) > 1 else deltas  # drop the compile-spanning delta
    steady_sorted = sorted(steady)
    median = steady_sorted[len(steady_sorted) // 2]

    result = {
        "model": model, "task": task, "batch": batch,
        "n_steps_timed": len(steady),
        "steady_s_per_step": median,
        "first_delta_excluded_s_per_step": deltas[0] if deltas else None,
    }
    print(f"[measure] RESULT: {result}", flush=True)
    return result


@app.local_entrypoint()
def run(model: str = "trm", task: str = "sudoku", step_cap: int = 100,
        batch: int = 768):
    """Launch a throughput sample. See module docstring for the exact commands."""
    import json
    res = measure.remote(model=model, task=task, step_cap=step_cap, batch=batch)
    print("\n" + json.dumps(res, indent=2))
