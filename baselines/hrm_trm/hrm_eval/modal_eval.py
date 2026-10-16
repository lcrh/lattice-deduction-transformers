"""Evaluate HRM checkpoints on Maze-Hard (exact and lenient) and Sudoku-Extreme (exact).

Uses HRM's own model and ACT loop (https://github.com/sapientinc/HRM, pinned
commit). Two build-only dependencies are stubbed so nothing compiles:
flash_attn becomes a torch SDPA shim (identical attention) and adam_atan2
becomes AdamW (eval never steps the optimizer). Scoring is in ../maze_scoring.py;
see baselines/README.md for the protocol.

Usage:
    uv run modal run baselines/hrm_trm/hrm_eval/modal_eval.py            # maze, released ckpt
    uv run modal run baselines/hrm_trm/hrm_eval/modal_eval.py::sudoku    # sudoku, released ckpt, first 10k of sudoku_test_perm
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

HRM_REMOTE = "/root/HRM"
CKPT_REPO = "sapientinc/HRM-checkpoint-maze-30x30-hard"
SUDOKU_CKPT_REPO = "sapientinc/HRM-checkpoint-sudoku-extreme"
# Pinned upstream commit, cloned at image build.
HRM_COMMIT = "ac15626f8db096a63c775b84c9dc868776a6feda"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "numpy", "einops", "tqdm", "coolname", "pydantic",
        "argdantic", "omegaconf", "hydra-core", "huggingface_hub", "wandb",
        "pyyaml",
    )
    .env({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "8"})
    .run_commands(
        f"git clone https://github.com/sapientinc/HRM.git {HRM_REMOTE}",
        f"cd {HRM_REMOTE} && git checkout {HRM_COMMIT}",
    )
    .add_local_file(Path(__file__).parent.parent / "sudoku_rows.py", "/root/sudoku_rows.py")
    .add_local_file(Path(__file__).parent.parent / "maze_scoring.py", "/root/maze_scoring.py")
)

app = modal.App("hrm-eval")

# LDT volumes: the Sudoku cache (index-alignment check) and the eval root
# where per-puzzle rows are written.
ldt_data = modal.Volume.from_name("lattice-diffusion-data", create_if_missing=True)
ldt_ckpt = modal.Volume.from_name("lattice-diffusion-checkpoints", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("hrm-maze-aug-ckpts", create_if_missing=True)


def _install_stubs():
    """Inject fake `flash_attn` / `adam_atan2` modules so HRM imports without
    any CUDA-extension build. Must run before importing HRM's `pretrain`."""
    import types

    import torch
    import torch.nn.functional as F

    def sdpa_flash_attn_func(q, k, v, causal=False, **kwargs):
        # q,k,v: [batch, seq, heads, head_dim]; SDPA wants [batch, heads, seq, d]
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
        if kt.shape[1] != qt.shape[1]:  # GQA: repeat kv heads
            rep = qt.shape[1] // kt.shape[1]
            kt = kt.repeat_interleave(rep, dim=1)
            vt = vt.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return out.transpose(1, 2)

    fa = types.ModuleType("flash_attn")
    fa.flash_attn_func = sdpa_flash_attn_func
    sys.modules["flash_attn"] = fa  # flash_attn_interface stays absent -> ImportError -> this

    class _AdamATan2(torch.optim.AdamW):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, **kw):
            super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)

    aa = types.ModuleType("adam_atan2")
    aa.AdamATan2 = _AdamATan2
    sys.modules["adam_atan2"] = aa


def _load_weights(model, checkpoint: str) -> None:
    """Load a state dict, dropping torch.compile's `_orig_mod.` prefix if present."""
    import torch
    state = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()}, assign=True)


@app.function(image=image, gpu="H100", timeout=3600, volumes={"/vol": ckpt_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run(checkpoint: str = "") -> dict:
    """Maze scores of the released checkpoint ("") or a /vol/.../step_N with all_config.yaml."""
    _install_stubs()  # before importing HRM

    os.chdir(HRM_REMOTE)
    sys.path[:0] = [HRM_REMOTE, "/root"]

    import subprocess

    import torch
    import yaml
    from huggingface_hub import hf_hub_download

    import maze_scoring

    # 1) build the maze test/train dataset HRM expects at data/maze-30x30-hard-1k
    print("[hrm] building maze dataset …", flush=True)
    subprocess.run([sys.executable, "dataset/build_maze_dataset.py"], cwd=HRM_REMOTE, check=True)

    # 2) resolve checkpoint + config (released HF, or a local volume path)
    out_dir = "/root/out"
    os.makedirs(out_dir, exist_ok=True)
    if checkpoint:  # local checkpoint from our training volume
        ckpt = checkpoint
        cfg_path = os.path.join(os.path.dirname(checkpoint), "all_config.yaml")
        print(f"[hrm] evaluating local checkpoint: {ckpt}", flush=True)
    else:           # released reproduction checkpoint
        ckpt_dir = "/root/ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = hf_hub_download(CKPT_REPO, "checkpoint", local_dir=ckpt_dir)
        hf_hub_download(CKPT_REPO, "all_config.yaml", local_dir=ckpt_dir)
        cfg_path = os.path.join(ckpt_dir, "all_config.yaml")

    from pretrain import PretrainConfig, create_dataloader, evaluate, init_train_state

    with open(cfg_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))
    config.eval_save_outputs = ["inputs", "labels", "logits"]
    config.checkpoint_path = out_dir

    train_loader, train_md = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    eval_loader, eval_md = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)

    train_state = init_train_state(config, train_md, world_size=1)
    _load_weights(train_state.model, ckpt)
    train_state.step = 0
    train_state.model.eval()

    print("[hrm] running HRM ACT inference over maze test set …", flush=True)
    metrics = evaluate(config, train_state, eval_loader, eval_md, rank=0, world_size=1)
    print(f"[hrm] official metrics: {metrics}", flush=True)

    # 3) score the saved predictions
    preds = torch.load(os.path.join(out_dir, "step_0_all_preds.0"))
    scores = maze_scoring.score_mazes(preds["inputs"].numpy(), preds["labels"].numpy(),
                                      preds["logits"].float().numpy().argmax(-1))
    maze_scoring.print_scores("HRM", scores)
    return scores


@app.local_entrypoint()
def main(checkpoint: str = ""):
    print(run.remote(checkpoint))


@app.function(image=image, volumes={"/vol": ckpt_vol}, timeout=300)
def _list_ckpts(run_dir: str) -> list[str]:
    import glob
    fs = [f for f in glob.glob(os.path.join("/vol", run_dir, "step_*"))
          if os.path.basename(f).removeprefix("step_").isdigit()]
    return sorted(fs, key=lambda f: int(os.path.basename(f).removeprefix("step_")))


@app.local_entrypoint()
def sweep(run_dir: str, out: str = "baselines/hrm_trm/results/hrm_sweep.json"):
    """Score every checkpoint of a training run on the volume (exact and lenient)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import maze_scoring

    ckpts = _list_ckpts.remote(run_dir)
    print(f"[sweep] {len(ckpts)} checkpoints in {run_dir}", flush=True)
    maze_scoring.write_sweep(ckpts, list(run.map(ckpts)), out)


@app.function(image=image, gpu="B200", timeout=6 * 3600, memory=32768,
              volumes={"/ldt_data": ldt_data, "/ldt_ckpt": ldt_ckpt},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_sudoku(indices: list[int], checkpoint: str = "", batch_size: int = 0,
               tag: str = "sudoku_test_perm") -> dict:
    """Exact-match accuracy on the given Sudoku-Extreme test indices, in order.

    checkpoint="" is the authors' release. Writes per-puzzle rows and a summary
    under eval/baselines/hrm/<tag>/ on the LDT checkpoint volume.
    """
    import subprocess

    _install_stubs()
    os.chdir(HRM_REMOTE)
    sys.path[:0] = [HRM_REMOTE, "/root"]

    import numpy as np
    import torch
    import yaml
    from huggingface_hub import hf_hub_download

    import sudoku_rows

    if checkpoint:
        ckpt = checkpoint
        cfg_path = os.path.join(os.path.dirname(checkpoint), "all_config.yaml")
    else:
        ckpt_dir = "/root/ckpt_sudoku"
        ckpt = hf_hub_download(SUDOKU_CKPT_REPO, "checkpoint", local_dir=ckpt_dir)
        hf_hub_download(SUDOKU_CKPT_REPO, "all_config.yaml", local_dir=ckpt_dir)
        cfg_path = os.path.join(ckpt_dir, "all_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # The train split is built (unaugmented) only because create_dataloader /
    # init_train_state need its metadata; the test split is then cut down to
    # `indices`.
    print(f"[hrm] building sudoku dataset at {cfg['data_path']} …", flush=True)
    subprocess.run([sys.executable, "dataset/build_sudoku_dataset.py",
                    "--output-dir", cfg["data_path"], "--subsample-size", "1000",
                    "--num-aug", "0"], cwd=HRM_REMOTE, check=True)
    test_dir = os.path.join(cfg["data_path"], "test")
    full_inputs = sudoku_rows.subset_test_split(test_dir, indices)
    aligned = sudoku_rows.check_alignment(full_inputs, "/ldt_data/sudoku_extreme_test.pt")
    if aligned is False:
        raise RuntimeError("HRM test split does not match the LDT cache index order")

    from pretrain import PretrainConfig, create_dataloader, init_train_state

    config = PretrainConfig(**cfg)
    if batch_size:
        config.global_batch_size = batch_size
    _, train_md = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    eval_loader, _ = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)

    train_state = init_train_state(config, train_md, world_size=1)
    _load_weights(train_state.model, ckpt)
    train_state.model.eval()

    # Same loop as HRM's evaluate(), keeping argmax predictions instead of
    # the full logits.
    print(f"[hrm] running HRM ACT inference over {len(indices)} sudoku test puzzles …",
          flush=True)
    t_eval = time.time()
    inputs, labels, preds = [], [], []
    with torch.inference_mode():
        for _, batch, _ in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
            while True:
                carry, _, _, out, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=["logits"])
                if all_finish:
                    break
            inputs.append(batch["inputs"].cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            preds.append(out["logits"].argmax(-1).cpu().numpy())
    torch.cuda.synchronize()
    eval_seconds = time.time() - t_eval

    inputs, labels, preds = (np.concatenate(a) for a in (inputs, labels, preds))
    keep = sudoku_rows.real_rows(inputs)
    inputs, labels, preds = inputs[keep], labels[keep], preds[keep]
    if not np.array_equal(inputs, np.load(os.path.join(test_dir, "all__inputs.npy"))):
        raise RuntimeError("eval order does not match the subset test split order")

    rows = sudoku_rows.build_rows(indices, labels, preds)
    summary = sudoku_rows.summarize(
        rows, model="HRM",
        checkpoint=ckpt if checkpoint else f"hf://{SUDOKU_CKPT_REPO}/checkpoint",
        hrm_commit=HRM_COMMIT, index_list="sudoku_test_perm", gpu="B200", n_gpus=1,
        global_batch_size=config.global_batch_size,
        eval_seconds=eval_seconds, gpu_seconds=eval_seconds,
        index_aligned_with_ldt_cache=aligned,
    )
    sudoku_rows.write_outputs(f"/ldt_ckpt/eval/baselines/hrm/{tag}", rows, summary)
    ldt_ckpt.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def sudoku(n: int = 10_000, checkpoint: str = "", batch_size: int = 0,
           out: str = "baselines/hrm_trm/results/hrm_sudoku.json"):
    """Evaluate on the first n puzzles of sudoku_test_perm (LDT's ordering)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sudoku_rows

    indices = sudoku_rows.first_n_of_test_perm(n)
    summary = sudoku_rows.add_interval(
        run_sudoku.remote(indices, checkpoint, batch_size, f"sudoku_test_perm_n{n}"))
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}", flush=True)
