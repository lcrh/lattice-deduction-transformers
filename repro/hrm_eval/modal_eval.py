"""Evaluate the official HRM maze-30x30-hard checkpoint with LENIENT scoring.

HRM (https://github.com/sapientinc/HRM, vendored as the `HRM/` submodule) reports
`exact_accuracy`: the predicted grid must match the dataset's ONE labelled
shortest path cell-for-cell. Mazes have many shortest paths, so that under-counts
correct solutions. Here we additionally compute a **lenient** score: a prediction
is correct if its marked `o`-cells (plus S/G) form a valid simple path from start
to goal whose length equals the maze's optimal (BFS) shortest-path length — i.e.
*any* optimal path counts.

We reuse HRM's own model + ACT inference loop (their `evaluate()`); two build-only
dependencies are stubbed at runtime so nothing has to compile:
  * `flash_attn.flash_attn_func` -> a torch SDPA shim (mathematically identical
    exact attention; the maze model is non-causal).
  * `adam_atan2.AdamATan2` -> `torch.optim.AdamW` (eval never steps the optimizer;
    the optimizer is only constructed by `init_train_state`).

Usage:
    uv run modal run experiments/hrm_eval/modal_eval.py
"""

import os
from collections import deque

import modal

HRM_REMOTE = "/root/HRM"
CKPT_REPO = "sapientinc/HRM-checkpoint-maze-30x30-hard"
# Pinned upstream HRM commit (https://github.com/sapientinc/HRM). Cloned fresh at
# image-build time.
# (flash-attn, adam-atan2) are stubbed at runtime in `_install_stubs()`.
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
)

app = modal.App("hrm-maze-lenient-eval")

# maze token ids (build_maze_dataset.py: CHARSET="# SGo" -> ids 1..5, PAD=0)
WALL, EMPTY, START, GOAL, PATH = 1, 2, 3, 4, 5


def _install_stubs():
    """Inject fake `flash_attn` / `adam_atan2` modules so HRM imports without
    any CUDA-extension build. Must run before importing HRM's `pretrain`."""
    import sys
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


def _bfs_dist(passable, src, dst):
    """Shortest 4-connected path length (in steps) over passable cells."""
    H, W = passable.shape
    seen = {src}
    dq = deque([(src, 0)])
    while dq:
        (r, c), d = dq.popleft()
        if (r, c) == dst:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                dq.append(((nr, nc), d + 1))
    return None  # unreachable


def _lenient_ok(inp, pred):
    """inp, pred: [H, W] int grids. True iff pred's path cells form a simple
    S->G path of OPTIMAL length over the maze's passable cells."""
    import numpy as np

    passable = inp != WALL
    s = np.argwhere(inp == START)
    g = np.argwhere(inp == GOAL)
    if len(s) != 1 or len(g) != 1:
        return False
    src, dst = tuple(s[0]), tuple(g[0])

    P = set(map(tuple, np.argwhere((pred == START) | (pred == GOAL) | (pred == PATH))))
    if src not in P or dst not in P:
        return False
    if any(not passable[r, c] for (r, c) in P):  # path crosses a wall
        return False

    def nbrs(cell):
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n in P:
                yield n

    deg = {c: sum(1 for _ in nbrs(c)) for c in P}
    # simple path: endpoints degree 1, interior degree 2
    if deg.get(src) != 1 or deg.get(dst) != 1:
        return False
    if any(deg[c] != 2 for c in P if c not in (src, dst)):
        return False
    # connected (single component covering all path cells)
    seen = {src}
    dq = deque([src])
    while dq:
        x = dq.popleft()
        for n in nbrs(x):
            if n not in seen:
                seen.add(n)
                dq.append(n)
    if len(seen) != len(P):
        return False
    # optimal: path length (cells-1) == shortest distance
    return (len(P) - 1) == _bfs_dist(passable, src, dst)


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={"/vol": modal.Volume.from_name("hrm-maze-aug-ckpts", create_if_missing=True)},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run(checkpoint: str = ""):
    """checkpoint="" -> the released HF checkpoint; else a local path (e.g. a
    /vol/.../step_N from our augmented training, with all_config.yaml beside it)."""
    import sys

    _install_stubs()  # before importing HRM

    os.chdir(HRM_REMOTE)
    sys.path.insert(0, HRM_REMOTE)

    import subprocess

    import numpy as np
    import torch
    import yaml
    from huggingface_hub import hf_hub_download

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
    try:
        train_state.model.load_state_dict(torch.load(ckpt, map_location="cuda"), assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in torch.load(ckpt, map_location="cuda").items()},
            assign=True)
    train_state.step = 0
    train_state.model.eval()

    print("[hrm] running HRM ACT inference over maze test set …", flush=True)
    metrics = evaluate(config, train_state, eval_loader, eval_md, rank=0, world_size=1)
    print(f"[hrm] official metrics: {metrics}", flush=True)

    # 3) load saved predictions and score leniently
    preds = torch.load(os.path.join(out_dir, "step_0_all_preds.0"))
    inputs = preds["inputs"].numpy()
    labels = preds["labels"].numpy()
    logits = preds["logits"].float().numpy()
    pred_ids = logits.argmax(-1)

    side = int(round(inputs.shape[1] ** 0.5))
    n_real = n_exact = n_lenient = 0
    for i in range(inputs.shape[0]):
        inp = inputs[i].reshape(side, side)
        if not ((inp == START).any() and (inp == GOAL).any()):
            continue  # padded / blank row
        n_real += 1
        lab = labels[i].reshape(side, side)
        prd = pred_ids[i].reshape(side, side)
        if np.array_equal(prd, lab):
            n_exact += 1
        if _lenient_ok(inp, prd):
            n_lenient += 1

    print("\n" + "=" * 56, flush=True)
    print(f"HRM maze-30x30-hard  ({n_real} test puzzles)", flush=True)
    print("=" * 56, flush=True)
    print(f"  exact   accuracy: {n_exact}/{n_real} = {100*n_exact/n_real:.2f}%", flush=True)
    print(f"  lenient accuracy: {n_lenient}/{n_real} = {100*n_lenient/n_real:.2f}%", flush=True)
    print("=" * 56, flush=True)
    return {"n": n_real, "exact": n_exact, "lenient": n_lenient}


@app.local_entrypoint()
def main(checkpoint: str = ""):
    print(run.remote(checkpoint))


@app.function(image=image, volumes={"/vol": modal.Volume.from_name("hrm-maze-aug-ckpts", create_if_missing=True)}, timeout=300)
def _list_ckpts(run_dir: str):
    import glob
    fs = [f for f in glob.glob(os.path.join("/vol", run_dir, "step_*"))
          if os.path.basename(f).removeprefix("step_").isdigit()]
    return sorted(fs, key=lambda f: int(os.path.basename(f).removeprefix("step_")))


@app.local_entrypoint()
def sweep(run_dir: str = "Maze-30x30-hard-1k ACT-torch/HierarchicalReasoningModel_ACTV1 pistachio-myna",
          out: str = "repro/results/hrm_aug_sweep.json"):
    """Eval EVERY checkpoint in a run dir (exact+lenient) -> JSON for plotting."""
    import json
    ckpts = _list_ckpts.remote(run_dir)
    print(f"[sweep] {len(ckpts)} checkpoints in {run_dir}", flush=True)
    results = list(run.map(ckpts))
    rows = sorted(
        [{"step": int(os.path.basename(c).removeprefix("step_")), **r}
         for c, r in zip(ckpts, results)],
        key=lambda x: x["step"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[sweep] wrote {out}", flush=True)
    for r in rows:
        print(f"  step {r['step']:>7}: exact={100*r['exact']/r['n']:.2f}%  "
              f"lenient={100*r['lenient']/r['n']:.2f}%", flush=True)
