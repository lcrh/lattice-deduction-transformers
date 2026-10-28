"""Sudoku-Extreme training and inference throughput of Equilibrium Reasoners on one B200.

EqR (Huang, Geng & Kolter, ICML 2026) publishes no training time.
`throughput` times its released Sudoku-Extreme training config
(`config/train/eqr_sudoku.yaml`), without checkpoints or W&B, as the
difference between a short and a long run.
`inference` times the released checkpoint under the paper's full evaluation
config (depth 64, breadth 128) as the difference between a short and a long
run, so startup and compilation cancel out.

Usage:
    uv run modal run baselines/eqr/modal_throughput.py::run_throughput
    uv run modal run baselines/eqr/modal_throughput.py::run_inference
"""

import subprocess

import modal

EQR_REMOTE = "/root/EqR"
EQR_COMMIT = "134c7541711125860954c78463023c583b2ef21b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "pydantic", "omegaconf", "hydra-core", "wandb==0.19.11", "coolname",
                 "psutil", "tqdm", "pyyaml", "argdantic", "colorama", "huggingface_hub",
                 "exceptiongroup", "matplotlib", "plotly",
                 "setuptools", "wheel", "ninja", "packaging")
    .env({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "8",
          "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0"})
    # adam-atan2's setup.py lists no sm_100 arch; add it, as for HRM.
    .run_commands(
        "pip download adam-atan2==0.0.3 --no-deps --no-binary :all: -d /tmp/aa",
        "tar xzf /tmp/aa/adam_atan2-*.tar.gz -C /tmp",
        'sed -i \'s/"90"}/"90", "100"}/\' /tmp/adam_atan2-*/setup.py',
        "CC=gcc CXX=g++ CUDAHOSTCXX=/usr/bin/g++ NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++' "
        "TORCH_CUDA_ARCH_LIST=10.0 pip install /tmp/adam_atan2-*/ --no-build-isolation",
        f"git clone https://github.com/locuslab/EqR.git {EQR_REMOTE}",
        f"cd {EQR_REMOTE} && git checkout {EQR_COMMIT}",
    )
)

app = modal.App("eqr-sudoku-throughput")


@app.function(image=image, gpu="B200", timeout=3 * 3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def throughput(short_steps: int = 200, long_steps: int = 1200) -> dict:
    """Steady-state s/step of `scripts/train.sh eqr_sudoku` on one B200.

    EqR only shows a training progress bar with W&B enabled, so this times two
    runs capped at different step counts; startup, data loading and the
    closing evaluation are the same in both and cancel.
    """
    import sys
    import time

    subprocess.run([sys.executable, "-m", "dataset.build_sudoku_dataset",
                    "--output-dir", "data/sudoku-extreme-1k-aug-1000",
                    "--subsample-size", "1000", "--num-aug", "1000", "--seed", "42"],
                   cwd=EQR_REMOTE, check=True)
    seconds = {}
    for steps in (short_steps, long_steps):
        t0 = time.perf_counter()
        subprocess.run(["bash", "scripts/train.sh", "eqr_sudoku", "+wandb_mode=disabled",
                        f"++max_steps={steps}", "eval_interval_steps=100000000",
                        "checkpoint_interval_steps=100000000"],
                       cwd=EQR_REMOTE, check=True)
        seconds[steps] = time.perf_counter() - t0
    return {"model": "EqR", "seconds": seconds,
            "s_per_step": (seconds[long_steps] - seconds[short_steps]) / (long_steps - short_steps)}


@app.function(image=image, gpu="B200", timeout=4 * 3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def inference(short_steps: int = 2, long_steps: int = 12, batch: int = 96) -> dict:
    """B200 seconds per puzzle of the released checkpoint at depth 64, breadth 128."""
    import time

    subprocess.run(["bash", "scripts/download_artifacts.sh", "--with-ckpts"],
                   cwd=EQR_REMOTE, check=True)
    seconds = {}
    for steps in (short_steps, long_steps):
        t0 = time.perf_counter()
        subprocess.run(["python", "evaluate.py", "eval_yaml=config/eval/sudoku_full_noise05.yaml",
                        "checkpoint=downloaded_checkpoints/sudoku-extreme/eqr.pth",
                        f"global_batch_size={batch}", f"max_eval_steps={steps}",
                        "+wandb_mode=disabled"],
                       cwd=EQR_REMOTE, check=True)
        seconds[steps] = time.perf_counter() - t0
    puzzles = (long_steps - short_steps) * batch
    return {"model": "EqR", "setting": "depth 64, breadth 128", "puzzles_timed": puzzles,
            "seconds": seconds, "s_per_puzzle": (seconds[long_steps] - seconds[short_steps]) / puzzles}


@app.local_entrypoint()
def run_throughput():
    print(throughput.remote())


@app.local_entrypoint()
def run_inference():
    print(inference.remote())
