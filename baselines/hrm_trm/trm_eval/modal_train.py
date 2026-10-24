"""TRM Maze-Hard training on a B200, with or without 8x dihedral augmentation.

Uses the real adam-atan2 optimizer (JIT-compiled in the CUDA-dev image). TRM
saves weights only and cannot resume, so a run is bounded by Modal's 24 h timeout.

Usage:
    uv run modal run baselines/hrm_trm/trm_eval/modal_train.py::run_train --aug
    uv run modal run baselines/hrm_trm/trm_eval/modal_train.py::run_train --no-aug
"""

import os
import subprocess

import modal

TRM_REMOTE = "/root/TRM"
TRM_COMMIT = "413f2f5c290e4091fda8efd73c7a2b3e329e1527"

# CUDA-dev base so torch.utils.cpp_extension can build adam-atan2 (nvcc).
# cu128 + sm_100 for Blackwell/B200.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "einops", "tqdm", "coolname", "pydantic", "argdantic",
        "wandb", "omegaconf", "hydra-core", "huggingface_hub", "packaging",
        "numba", "ninja",
    )
    .env({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "8",
          "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0"})
    .run_commands(
        f"git clone https://github.com/alphaXiv/TinyRecursiveModels.git {TRM_REMOTE}",
        f"cd {TRM_REMOTE} && git checkout {TRM_COMMIT}",
        f"cd {TRM_REMOTE} && pip install -e . --no-deps",
    )
    .add_local_file(os.path.join(os.path.dirname(__file__), "..", "tqdm_rate.py"),
                    "/root/tqdm_rate.py")
)

app = modal.App("trm-maze-train")

# checkpoints persist here (train.py writes checkpoints/<project>/<run>/step_N)
CKPT_DIR = f"{TRM_REMOTE}/checkpoints"
ckpt_vol = modal.Volume.from_name("trm-maze-aug-ckpts", create_if_missing=True)

# maze training recipe (TRM README, single-GPU maze) + larger batch for B200.
TRAIN_ARGS = [
    "arch=trm",
    "data_paths=[data/maze-30x30-hard-1k]",
    "evaluators=[]",
    "epochs=50000", "eval_interval=5000",
    "lr=2e-4", "puzzle_emb_lr=1e-4",
    "weight_decay=1.0", "puzzle_emb_weight_decay=1.0",
    # Keep the arch overrides in sync with cfg_maze.yaml (used for eval).
    "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=4",
    "global_batch_size=768", "lr_warmup_steps=4000",
    "checkpoint_every_eval=True", "ema=True",
]


@app.function(image=image, gpu="B200", timeout=86400,  # 24h = Modal max
              volumes={CKPT_DIR: ckpt_vol},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")])  # WANDB_API_KEY
def train(epochs: int = 50000, aug: bool = True, run_name: str = "",
          project: str = "maze-hard-repro"):
    """TRM maze training at the published recipe (about 26 h, so it stops at the 24 h cap).

    `aug` toggles 8x dihedral augmentation. Checkpoints go to
    checkpoints/<project>/<run_name> on the volume, committed every 30 min.
    """
    import sys
    import threading

    run_name = run_name or f"trm-maze-{'aug' if aug else 'noaug'}"
    build_cmd = [sys.executable, "-m", "trm.data.build_maze_dataset",
                 "--output-dir", "data/maze-30x30-hard-1k"]
    if aug:
        build_cmd.append("--aug")
    print(f"[trm] building maze dataset (aug={aug}) …", flush=True)
    subprocess.run(build_cmd, cwd=TRM_REMOTE, check=True)

    stop = threading.Event()

    def _committer():
        while not stop.wait(1800):  # every 30 min
            try:
                ckpt_vol.commit()
                print("[trm] (volume committed)", flush=True)
            except Exception as e:
                print(f"[trm] commit error: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    args = ([a for a in TRAIN_ARGS if not a.startswith("epochs=")]
            + [f"epochs={epochs}", f"+run_name={run_name}", f"+project_name={project}"])
    cmd = [sys.executable, "scripts/train.py", *args]
    # stream combined output to BOTH the Modal log and a persistent logfile on
    # the volume (committed periodically alongside checkpoints). wandb logs
    # online via WANDB_API_KEY from the wandb-secret.
    logpath = os.path.join(CKPT_DIR, f"{run_name}.train.log")
    env = {**os.environ, "WANDB_MODE": "online"}  # enable wandb (key from secret)
    print(f"[trm] TRAIN (run_name={run_name}) — log -> {logpath}\n  {' '.join(cmd)}", flush=True)
    with open(logpath, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=TRM_REMOTE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
        rc = proc.wait()

    stop.set()
    ckpt_vol.commit()
    import glob
    cks = sorted(glob.glob(f"{CKPT_DIR}/**/step_*", recursive=True))
    print(f"[trm] train.py exited rc={rc}; {len(cks)} checkpoints committed", flush=True)
    print(f"[trm] latest: {cks[-3:]}", flush=True)
    return {"rc": rc, "n_ckpts": len(cks), "latest": cks[-1] if cks else None}


# Sudoku-Extreme recipe of the released checkpoints (TRM README), per variant.
SUDOKU_ARGS = [
    "arch=trm", "data_paths=[data/sudoku-extreme-1k-aug-1000]", "evaluators=[]",
    "epochs=50000", "eval_interval=50000", "lr=1e-4", "puzzle_emb_lr=1e-4",
    "weight_decay=1.0", "puzzle_emb_weight_decay=1.0", "global_batch_size=1536",
    "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=6", "ema=True",
]
SUDOKU_VARIANTS = {"mlp": ["arch.mlp_t=True", "arch.pos_encodings=none"], "att": []}


# A separate app: training needs a W&B secret, measuring throughput does not.
throughput_app = modal.App("trm-sudoku-throughput")


@throughput_app.function(image=image, gpu="B200", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def throughput(variant: str = "mlp", minutes: int = 12) -> dict:
    """Steady-state Sudoku-Extreme training s/step on one B200 (no checkpoints)."""
    import sys

    from tqdm_rate import seconds_per_step

    subprocess.run([sys.executable, "-m", "trm.data.build_sudoku_dataset",
                    "--output-dir", "data/sudoku-extreme-1k-aug-1000",
                    "--subsample-size", "1000", "--num-aug", "1000"],
                   cwd=TRM_REMOTE, check=True)
    cmd = ["timeout", str(minutes * 60), sys.executable, "scripts/train.py",
           *SUDOKU_ARGS, *SUDOKU_VARIANTS[variant], f"+run_name=throughput-{variant}"]
    out = subprocess.run(cmd, cwd=TRM_REMOTE, capture_output=True, text=True)
    step, total, s_per_step = seconds_per_step(out.stdout + out.stderr)
    return {"model": f"TRM-{variant}", "step": step, "total_steps": total, "s_per_step": s_per_step}


@throughput_app.local_entrypoint()
def run_throughput(variant: str = "mlp", minutes: int = 12):
    print(throughput.remote(variant=variant, minutes=minutes))


@app.local_entrypoint()
def run_train(aug: bool = True, epochs: int = 50000, run_name: str = "",
              project: str = "maze-hard-repro"):
    print(train.remote(epochs=epochs, aug=aug, run_name=run_name, project=project))
