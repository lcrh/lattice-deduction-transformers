"""Training for HRM maze-30x30-hard on B200, with/without 8x dihedral augmentation.

HRM requires flash-attn (no fallback) and adam-atan2. We drop a flash_attn shim
onto PYTHONPATH so the `pretrain.py` subprocess picks it up:
  * flash_attn.py -> flash_attn_func = torch SDPA (exact attention; REQUIRED to
    run at all on B200 since flash-attn has no Blackwell wheels).
The real adam-atan2 CUDA kernel is built into `train_image` (see below).

Usage:
    uv run modal run repro/hrm_eval/modal_train.py::run_train --aug
    uv run modal run repro/hrm_eval/modal_train.py::run_train --no-aug
    uv run modal run repro/hrm_eval/modal_train.py::check   # smoke-test adam-atan2
"""

import os
import subprocess

import modal

HRM_REMOTE = "/root/HRM"
HRM_COMMIT = "ac15626f8db096a63c775b84c9dc868776a6feda"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    # exclusive index_url forces a cu128 torch build that MATCHES the 12.8 nvcc
    # in the devel base (additive extra_index_url leaks in a cu13.0 torch from
    # PyPI, which breaks adam-atan2's nvcc==torch.cuda build check).
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "einops", "tqdm", "coolname", "pydantic", "argdantic",
        "wandb", "omegaconf", "hydra-core", "huggingface_hub", "pyyaml",
    )
    .env({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "8",
          "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0"})
    .run_commands(
        f"git clone https://github.com/sapientinc/HRM.git {HRM_REMOTE}",
        f"cd {HRM_REMOTE} && git checkout {HRM_COMMIT}",
    )
)

app = modal.App("hrm-maze-train")

# real training image: build the actual adam-atan2 CUDA optimizer (sdist).
# --no-build-isolation so torch is visible during the build and setup.py's
# CUDAExtension actually compiles `adam_atan2_backend` for sm_100 (a plain
# `pip install` builds a pure-python wheel with no backend -> ImportError).
train_image = (
    image
    .pip_install("setuptools", "wheel", "ninja", "packaging")
    # Build the REAL adam-atan2 kernel for Blackwell. Its setup.py hardcodes
    # NVIDIA_SUPPORTED_ARCHS = {"80","86","89","90"} and ignores
    # TORCH_CUDA_ARCH_LIST, so it never emits an sm_100 image -> "no kernel
    # image" on B200. We patch the arch set to add "100" and build from the
    # patched sdist (kernel/math unchanged; just compiled for the right GPU).
    # Also force g++ as nvcc host compiler (else it grabs a broken clang stub).
    .run_commands(
        "pip download adam-atan2 --no-deps --no-binary :all: -d /tmp/aa",
        "tar xzf /tmp/aa/adam_atan2-*.tar.gz -C /tmp",
        'sed -i \'s/"90"}/"90", "100"}/\' /tmp/adam_atan2-*/setup.py',
        "CC=gcc CXX=g++ CUDAHOSTCXX=/usr/bin/g++ NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++' "
        "TORCH_CUDA_ARCH_LIST=10.0 pip install /tmp/adam_atan2-*/ --no-build-isolation",
    )
)

HRM_CKPT_DIR = f"{HRM_REMOTE}/checkpoints"
hrm_vol = modal.Volume.from_name("hrm-maze-aug-ckpts", create_if_missing=True)

# HRM maze recipe (README), single-process; larger batch for B200.
TRAIN_ARGS = [
    "data_path=data/maze-30x30-hard-1k",
    "epochs=20000", "eval_interval=2000",
    "lr=1e-4", "puzzle_emb_lr=1e-4",
    "weight_decay=1.0", "puzzle_emb_weight_decay=1.0",
    "global_batch_size=768", "checkpoint_every_eval=True",
]

_FLASH_SHIM = '''
import torch
import torch.nn.functional as F
import torch.optim
# adam-atan2 0.0.3 calls a torch-internal that newer torch removed; restore it
# as a no-op (CUDA-graph safety hook, irrelevant to normal training / the
# optimizer math). This shim is imported by models.layers before any opt step.
if not hasattr(torch.optim.Optimizer, "_cuda_graph_capture_health_check"):
    torch.optim.Optimizer._cuda_graph_capture_health_check = lambda self: None
def flash_attn_func(q, k, v, causal=False, **kwargs):
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    if kt.shape[1] != qt.shape[1]:
        r = qt.shape[1] // kt.shape[1]
        kt = kt.repeat_interleave(r, dim=1); vt = vt.repeat_interleave(r, dim=1)
    return F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
'''

@app.function(image=train_image, gpu="B200", timeout=86400,  # 24h = Modal max
              volumes={HRM_CKPT_DIR: hrm_vol},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")])  # WANDB_API_KEY
def train(epochs: int = 120000, aug: bool = True, run_name: str = "",
          project: str = "maze-hard-repro"):
    """HRM maze training, extended past the 20k-epoch recipe (~26k steps / ~4h)
    to fill the 24h budget (~150k steps) so we capture the whole trajectory
    (recipe-point AND overtrained later steps). `aug` toggles 8x dihedral
    augmentation; checkpoints go to checkpoints/<project>/<run_name>. Uses the
    REAL adam-atan2 (built into train_image) + flash-attn->SDPA shim."""
    import sys
    import threading

    run_name = run_name or f"hrm-maze-{'aug' if aug else 'noaug'}"

    # only the flash-attn shim now (real adam-atan2 is installed in the image)
    os.makedirs("/root/shims", exist_ok=True)
    with open("/root/shims/flash_attn.py", "w") as f:
        f.write(_FLASH_SHIM)
    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/shims" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    build_cmd = [sys.executable, "dataset/build_maze_dataset.py",
                 "--output-dir", "data/maze-30x30-hard-1k"]
    if aug:
        build_cmd.append("--aug")
    print(f"[hrm] building maze dataset (aug={aug}) …", flush=True)
    subprocess.run(build_cmd, cwd=HRM_REMOTE, check=True, env=env)

    stop = threading.Event()

    def _committer():
        while not stop.wait(1800):
            try:
                hrm_vol.commit()
                print("[hrm] (volume committed)", flush=True)
            except Exception as e:
                print(f"[hrm] commit error: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    args = ([a for a in TRAIN_ARGS if not a.startswith("epochs=")]
            + [f"epochs={epochs}", f"+run_name={run_name}", f"+project_name={project}"])
    cmd = [sys.executable, "pretrain.py", *args]
    env["WANDB_MODE"] = "online"  # enable wandb (WANDB_API_KEY from wandb-secret)
    # stream combined output to BOTH the Modal log and a persistent logfile on
    # the volume (committed periodically alongside checkpoints).
    logpath = os.path.join(HRM_CKPT_DIR, f"{run_name}.train.log")
    print(f"[hrm] TRAIN (run_name={run_name}) — log -> {logpath}\n  {' '.join(cmd)}", flush=True)
    with open(logpath, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=HRM_REMOTE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lf.write(line); lf.flush()
            print(line, end="", flush=True)
        rc = proc.wait()

    stop.set()
    hrm_vol.commit()
    import glob
    cks = sorted(glob.glob(f"{HRM_CKPT_DIR}/**/step_*", recursive=True))
    print(f"[hrm] pretrain.py exited rc={rc}; {len(cks)} checkpoints committed", flush=True)
    print(f"[hrm] latest: {cks[-3:]}", flush=True)
    return {"rc": rc, "n_ckpts": len(cks), "latest": cks[-1] if cks else None}


def _patch_torch_optim_compat():
    """adam-atan2 0.0.3 was written against an older torch and calls
    Optimizer._cuda_graph_capture_health_check(), which newer torch removed.
    It's a CUDA-graph safety hook (no-op for normal training, which HRM uses) —
    restoring it as a no-op is an API-compat shim, NOT an optimizer change."""
    import torch.optim
    if not hasattr(torch.optim.Optimizer, "_cuda_graph_capture_health_check"):
        torch.optim.Optimizer._cuda_graph_capture_health_check = lambda self: None


@app.function(image=train_image, gpu="B200", timeout=300)
def check_optim():
    """Smoke-test the real adam-atan2 CUDA kernel on B200 before a 24h launch."""
    import torch
    _patch_torch_optim_compat()
    from adam_atan2 import AdamATan2
    p = torch.nn.Parameter(torch.randn(1024, 1024, device="cuda"))
    opt = AdamATan2([p], lr=1e-3)
    (p.square().mean()).backward()
    opt.step()
    torch.cuda.synchronize()
    print("[hrm] adam-atan2 step OK on", torch.cuda.get_device_name(0), flush=True)
    return "ok"


@app.local_entrypoint()
def check():
    print(check_optim.remote())


@app.local_entrypoint()
def run_train(aug: bool = True, epochs: int = 120000, run_name: str = "",
              project: str = "maze-hard-repro"):
    print(train.remote(epochs=epochs, aug=aug, run_name=run_name, project=project))
