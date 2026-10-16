"""Evaluate TRM checkpoints on Maze-Hard (exact and lenient) and Sudoku-Extreme (exact).

Drives TRM's own `run_eval_only.py` (alphaXiv/TinyRecursiveModels, pinned
commit); nothing compiles and no stubs are needed. The checkpoints are
alphaXiv's independent reproductions. Each model family needs its training
architecture config (cfg_maze.yaml, cfg_sudoku_{mlp,att}.yaml), because a
loop-count mismatch loads without error. Scoring is in ../maze_scoring.py;
see baselines/README.md for the protocol.

Usage:
    uv run modal run baselines/hrm_trm/trm_eval/modal_eval.py            # maze, released ckpt
    uv run modal run baselines/hrm_trm/trm_eval/modal_eval.py::sudoku    # sudoku TRM-MLP, first 10k of sudoku_test_perm
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

TRM_REMOTE = "/root/TRM"
CKPT = "alphaXiv/trm-model-maze/maze_hard_step_32550"  # username/repo/filename
SUDOKU_CKPTS = {  # variant -> (checkpoint, config name)
    "mlp": ("alphaXiv/trm-model-sudoku/step_32550_sudoku_epoch50k", "cfg_sudoku_mlp"),
    "att": ("alphaXiv/trm-model-sudoku/step_32550_sudoku_50k_epoch_attn_type", "cfg_sudoku_att"),
}
TRM_COMMIT = "413f2f5c290e4091fda8efd73c7a2b3e329e1527"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "numpy", "einops", "tqdm", "coolname", "pydantic",
        "argdantic", "wandb", "omegaconf", "hydra-core", "huggingface_hub",
        "packaging", "numba", "ninja",
    )
    .env({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "8", "DISABLE_COMPILE": "1"})
    .run_commands(
        f"git clone https://github.com/alphaXiv/TinyRecursiveModels.git {TRM_REMOTE}",
        f"cd {TRM_REMOTE} && git checkout {TRM_COMMIT}",
        f"cd {TRM_REMOTE} && pip install -e . --no-deps",
    )
    .add_local_file(Path(__file__).parent / "cfg_maze.yaml",
                    f"{TRM_REMOTE}/config/cfg_maze.yaml")
    .add_local_file(Path(__file__).parent / "cfg_sudoku_mlp.yaml",
                    f"{TRM_REMOTE}/config/cfg_sudoku_mlp.yaml")
    .add_local_file(Path(__file__).parent / "cfg_sudoku_att.yaml",
                    f"{TRM_REMOTE}/config/cfg_sudoku_att.yaml")
    .add_local_file(Path(__file__).parent.parent / "sudoku_rows.py", "/root/sudoku_rows.py")
    .add_local_file(Path(__file__).parent.parent / "maze_scoring.py", "/root/maze_scoring.py")
)

app = modal.App("trm-eval")

# LDT volumes: the Sudoku cache (index-alignment check) and the eval root
# where per-puzzle rows are written.
ldt_data = modal.Volume.from_name("lattice-diffusion-data", create_if_missing=True)
ldt_ckpt = modal.Volume.from_name("lattice-diffusion-checkpoints", create_if_missing=True)

# Our Maze retrains (mounted at /vol); pass --checkpoint /vol/<...>/step_N to evaluate one.
ckpt_vol = modal.Volume.from_name("trm-maze-aug-ckpts", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={"/vol": ckpt_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run(checkpoint: str = CKPT) -> dict:
    """Maze scores of one checkpoint (an HF path, or /vol/.../step_N)."""
    import glob
    import subprocess

    import torch

    sys.path.insert(0, "/root")
    import maze_scoring

    local = checkpoint.startswith("/vol/")
    with open(f"{TRM_REMOTE}/config/cfg_maze.yaml") as f:
        settings = {"model": "trm", "trm_commit": TRM_COMMIT, "config": f.read()}
    if local and (cached := maze_scoring.cached_scores(checkpoint, settings)):
        print(f"[trm] already scored: {checkpoint}", flush=True)
        return cached

    # 1) build the maze test/train dataset at data/maze-30x30-hard-1k
    print("[trm] building maze dataset …", flush=True)
    subprocess.run([sys.executable, "-m", "trm.data.build_maze_dataset"],
                   cwd=TRM_REMOTE, check=True)

    # 2) run TRM's own eval script (loads checkpoint from HF, EMA on by default,
    #    eval-only so no optimizer/adam-atan2 build). Saves preds to --outdir.
    outdir = "/root/trm_out"
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_eval_only.py",
        "--config", "config/cfg_maze.yaml",
        "--checkpoint", checkpoint,
        "--dataset", "data/maze-30x30-hard-1k",
        "--outdir", outdir,
        "--eval-save-outputs", "inputs", "labels", "logits", "preds",
        "--global-batch-size", "768",
    ]
    print(f"[trm] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=TRM_REMOTE, capture_output=True, text=True)
    print(proc.stdout[-6000:], flush=True)
    if proc.returncode != 0:
        print("[trm] stderr tail:\n" + proc.stderr[-4000:], flush=True)

    # 3) score the saved predictions
    files = sorted(glob.glob(os.path.join(outdir, "*_all_preds.0")))
    if not files:
        raise RuntimeError(f"no preds saved in {outdir}; see eval stdout above")
    preds = torch.load(files[-1], map_location="cpu")
    scores = maze_scoring.score_mazes(preds["inputs"].numpy(), preds["labels"].numpy(),
                                      preds["logits"].float().numpy().argmax(-1))
    maze_scoring.print_scores("TRM", scores)
    if local:
        maze_scoring.save_scores(checkpoint, settings, scores)
        ckpt_vol.commit()
    return scores


@app.function(image=image, volumes={"/vol": ckpt_vol}, timeout=300)
def _list_ckpts(run_dir: str) -> list[str]:
    import glob
    fs = [f for f in glob.glob(os.path.join("/vol", run_dir, "step_*"))
          if os.path.basename(f).removeprefix("step_").isdigit()]
    return sorted(fs, key=lambda f: int(os.path.basename(f).removeprefix("step_")))


@app.local_entrypoint()
def main(checkpoint: str = CKPT):
    print(run.remote(checkpoint))


@app.local_entrypoint()
def sweep(run_dir: str, out: str = "baselines/hrm_trm/results/trm_sweep.json"):
    """Score every checkpoint of a training run on the volume (exact and lenient)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import maze_scoring

    ckpts = _list_ckpts.remote(run_dir)
    print(f"[sweep] {len(ckpts)} checkpoints in {run_dir}", flush=True)
    maze_scoring.write_sweep(ckpts, list(run.map(ckpts)), out)


@app.function(image=image, gpu="B200", timeout=6 * 3600, memory=32768,
              volumes={"/ldt_data": ldt_data, "/ldt_ckpt": ldt_ckpt},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_sudoku(chunk: int, start: int, indices: list[int], out_dir: str, variant: str = "mlp",
               checkpoint: str = "", batch_size: int = 1536) -> None:
    """Exact-match results for one chunk of Sudoku-Extreme test indices, written to `out_dir`.

    Uses TRM's run_eval_only.py (EMA weights, bf16). checkpoint="" is the
    alphaXiv release for `variant`.
    """
    import glob
    import subprocess

    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    import sudoku_rows

    default_ckpt, config_name = SUDOKU_CKPTS[variant]
    checkpoint = checkpoint or default_ckpt
    data_dir = "data/sudoku-extreme-test"

    # The builder always writes train + test; train is unaugmented and unused.
    print("[trm] building sudoku dataset …", flush=True)
    subprocess.run([sys.executable, "-m", "trm.data.build_sudoku_dataset",
                    "--output-dir", data_dir, "--subsample-size", "1000", "--num-aug", "0"],
                   cwd=TRM_REMOTE, check=True)
    test_dir = os.path.join(TRM_REMOTE, data_dir, "test")
    full_inputs = sudoku_rows.subset_test_split(test_dir, indices)
    aligned = sudoku_rows.check_alignment(full_inputs, "/ldt_data/sudoku_extreme_test.pt")
    if aligned is False:
        raise RuntimeError("TRM test split does not match the LDT cache index order")

    outdir = f"/root/trm_sudoku_out/chunk{chunk:04d}"     # fresh per chunk on a reused container
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_eval_only.py",
        "--config", f"config/{config_name}.yaml",
        "--checkpoint", checkpoint,
        "--dataset", data_dir,
        "--outdir", outdir,
        "--eval-save-outputs", "inputs", "labels", "preds",
        "--global-batch-size", str(batch_size),
    ]
    print(f"[trm] {' '.join(cmd)}", flush=True)
    t_eval = time.time()
    proc = subprocess.run(cmd, cwd=TRM_REMOTE, capture_output=True, text=True)
    eval_seconds = time.time() - t_eval
    print(proc.stdout[-6000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError("run_eval_only.py failed:\n" + proc.stderr[-4000:])

    files = sorted(glob.glob(os.path.join(outdir, "*_all_preds.0")))
    if not files:
        raise RuntimeError(f"no preds saved in {outdir}; see eval stdout above")
    saved = torch.load(files[-1], map_location="cpu")
    inputs = saved["inputs"].numpy()
    labels = saved["labels"].numpy()
    pv = saved["preds"].numpy()
    preds = pv.argmax(-1) if pv.ndim == 3 else pv
    keep = sudoku_rows.real_rows(inputs)
    inputs, labels, preds = inputs[keep], labels[keep], preds[keep]
    if not np.array_equal(inputs, np.load(os.path.join(test_dir, "all__inputs.npy"))):
        raise RuntimeError("eval order does not match the subset test split order")

    rows = sudoku_rows.build_rows(indices, labels, preds, start)
    sudoku_rows.write_chunk(out_dir, chunk, rows, {
        "model": f"TRM-{variant}", "checkpoint": f"hf://{checkpoint}",
        "trm_commit": TRM_COMMIT, "config": config_name, "gpu": "B200", "n_gpus": 1,
        "global_batch_size": batch_size, "eval_seconds": eval_seconds,
        "index_aligned_with_ldt_cache": aligned,
    })
    ldt_ckpt.commit()
    print(f"[trm] chunk {chunk}: {sum(r['status'] == 'correct' for r in rows)}/{len(rows)}",
          flush=True)


@app.function(image=image, timeout=600, volumes={"/ldt_ckpt": ldt_ckpt})
def done_chunks(out_dir: str) -> set[int]:
    sys.path.insert(0, "/root")
    import sudoku_rows

    ldt_ckpt.reload()
    return sudoku_rows.done_chunks(out_dir)


@app.function(image=image, timeout=600, volumes={"/ldt_ckpt": ldt_ckpt})
def merge_chunks(out_dir: str, n_chunks: int) -> dict:
    sys.path.insert(0, "/root")
    import sudoku_rows

    ldt_ckpt.reload()
    summary = sudoku_rows.merge_chunks(out_dir, n_chunks, index_list="sudoku_test_perm")
    ldt_ckpt.commit()
    return summary


@app.local_entrypoint()
def sudoku(n: int = 10_000, variant: str = "mlp", checkpoint: str = "",
           batch_size: int = 1536, out: str = ""):
    """Evaluate on the first n puzzles of sudoku_test_perm (LDT's ordering); resumes by chunk."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sudoku_rows

    out_dir = f"/ldt_ckpt/eval/baselines/trm-{variant}/sudoku_test_perm_n{n}"
    parts = sudoku_rows.chunks(sudoku_rows.first_n_of_test_perm(n))
    done = done_chunks.remote(out_dir)
    todo = [(c, start, idx, out_dir, variant, checkpoint, batch_size)
            for c, start, idx in parts if c not in done]
    print(f"{len(parts) - len(todo)}/{len(parts)} chunks already done", flush=True)
    list(run_sudoku.starmap(todo))
    summary = sudoku_rows.add_interval(merge_chunks.remote(out_dir, len(parts)))
    out = out or f"baselines/hrm_trm/results/trm_{variant}_sudoku.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}", flush=True)
