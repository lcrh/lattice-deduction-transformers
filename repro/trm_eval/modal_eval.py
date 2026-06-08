"""Evaluate the TRM maze-30x30-hard checkpoint with LENIENT scoring.

TRM ("Tiny Recursive Models", https://github.com/alphaXiv/TinyRecursiveModels,
cloned at image-build time) reports `exact_accuracy`: the predicted grid must
match the dataset's ONE labelled shortest path cell-for-cell. Mazes have many
shortest paths, so that under-counts. We additionally compute a **lenient**
score: a prediction is correct if its marked `o`-cells (plus S/G) form a valid
simple path from start to goal whose length equals the maze's optimal (BFS)
shortest-path length — i.e. *any* optimal path counts.

The checkpoint `alphaXiv/trm-model-maze/maze_hard_step_32550` is an independent
reproduction (the original TRM authors released code only, no weights); their
report puts maze-hard exact_accuracy at 83.67% ± 2.28% (paper claim 85.3%).

Unlike HRM, TRM uses native `scaled_dot_product_attention` (no flash-attn) and
its adam-atan2 CUDA extension is only JIT-built when the optimizer is used —
`run_eval_only.py --eval-only` skips that — so NOTHING compiles and no runtime
stubs are needed. We drive their own `run_eval_only.py` and verify its saved
predictions.

Usage:
    uv run modal run experiments/trm_eval/modal_eval.py
"""

import os
from collections import deque

import modal

TRM_REMOTE = "/root/TRM"
CKPT = "alphaXiv/trm-model-maze/maze_hard_step_32550"  # username/repo/filename
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
)

app = modal.App("trm-maze-lenient-eval")

# our own augmented-training checkpoints live here (mounted read-only at /vol);
# pass --checkpoint /vol/<...>/step_N to eval one.
ckpt_vol = modal.Volume.from_name("trm-maze-aug-ckpts", create_if_missing=True)

# maze token ids (build_maze_dataset.py: CHARSET="# SGo" -> ids 1..5, PAD=0)
WALL, EMPTY, START, GOAL, PATH = 1, 2, 3, 4, 5


def _bfs_dist(passable, src, dst):
    """Shortest 4-connected path length (steps) over passable cells."""
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
    return None


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
    if any(not passable[r, c] for (r, c) in P):
        return False

    def nbrs(cell):
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n in P:
                yield n

    deg = {c: sum(1 for _ in nbrs(c)) for c in P}
    if deg.get(src) != 1 or deg.get(dst) != 1:
        return False
    if any(deg[c] != 2 for c in P if c not in (src, dst)):
        return False
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
    return (len(P) - 1) == _bfs_dist(passable, src, dst)


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={"/vol": ckpt_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run(checkpoint: str = CKPT):
    import glob
    import subprocess
    import sys

    import numpy as np
    import torch

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

    # 3) load saved preds and score leniently
    files = sorted(glob.glob(os.path.join(outdir, "*_all_preds.0")))
    if not files:
        raise RuntimeError(f"no preds saved in {outdir}; see eval stdout above")
    preds = torch.load(files[-1], map_location="cpu")
    print(f"[trm] preds keys: {list(preds.keys())}  shapes: "
          f"{ {k: tuple(v.shape) for k, v in preds.items()} }", flush=True)

    inputs = preds["inputs"].numpy()
    labels = preds["labels"].numpy()
    # predicted grid: prefer logits (argmax); else 'preds' (argmax if 3D else as-is)
    if "logits" in preds:
        pred_ids = preds["logits"].float().numpy().argmax(-1)
    else:
        pv = preds["preds"].numpy()
        pred_ids = pv.argmax(-1) if pv.ndim == 3 else pv

    side = int(round(inputs.shape[1] ** 0.5))
    n_real = n_exact = n_lenient = 0
    for i in range(inputs.shape[0]):
        inp = inputs[i].reshape(side, side)
        if not ((inp == START).any() and (inp == GOAL).any()):
            continue
        n_real += 1
        if np.array_equal(pred_ids[i].reshape(side, side), labels[i].reshape(side, side)):
            n_exact += 1
        if _lenient_ok(inp, pred_ids[i].reshape(side, side)):
            n_lenient += 1

    print("\n" + "=" * 56, flush=True)
    print(f"TRM maze-30x30-hard  ({n_real} test puzzles)", flush=True)
    print("=" * 56, flush=True)
    print(f"  exact   accuracy: {n_exact}/{n_real} = {100*n_exact/n_real:.2f}%", flush=True)
    print(f"  lenient accuracy: {n_lenient}/{n_real} = {100*n_lenient/n_real:.2f}%", flush=True)
    print("=" * 56, flush=True)
    return {"n": n_real, "exact": n_exact, "lenient": n_lenient}


@app.function(image=image, volumes={"/vol": ckpt_vol}, timeout=300)
def _list_ckpts(run_dir: str):
    import glob
    fs = [f for f in glob.glob(os.path.join("/vol", run_dir, "step_*"))
          if os.path.basename(f).removeprefix("step_").isdigit()]
    return sorted(fs, key=lambda f: int(os.path.basename(f).removeprefix("step_")))


@app.local_entrypoint()
def main(checkpoint: str = CKPT):
    print(run.remote(checkpoint))


@app.local_entrypoint()
def sweep(run_dir: str = "Maze-30x30-hard-1k-ACT-torch/maze_aug_probe",
          out: str = "repro/results/trm_aug_sweep.json"):
    """Eval EVERY checkpoint in a run dir (exact+lenient) and save a JSON of
    {step, exact, lenient} for plotting accuracy-over-training."""
    import json
    ckpts = _list_ckpts.remote(run_dir)
    print(f"[sweep] {len(ckpts)} checkpoints in {run_dir}", flush=True)
    results = list(run.map(ckpts))  # one container per checkpoint
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
