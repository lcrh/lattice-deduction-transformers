"""Evaluate TRM checkpoints: Maze-30x30-hard (with LENIENT scoring) and
Sudoku-Extreme (exact match on the first N puzzles of LDT's
`sudoku_test_perm` ordering; default N = 10,000).

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

Sudoku checkpoints (`alphaXiv/trm-model-sudoku`, also an independent
reproduction) need the matching architecture config: `cfg_sudoku_mlp.yaml`
(TRM-MLP, the variant behind the paper's 87.4%) or `cfg_sudoku_att.yaml`.

Usage:
    uv run modal run baselines/hrm_trm/trm_eval/modal_eval.py            # maze, released ckpt
    uv run modal run baselines/hrm_trm/trm_eval/modal_eval.py::sudoku    # sudoku TRM-MLP, first 10k of sudoku_test_perm
"""

import json
import os
import time
from collections import deque
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
    .add_local_file(Path(__file__).parent / "cfg_sudoku_mlp.yaml",
                    f"{TRM_REMOTE}/config/cfg_sudoku_mlp.yaml")
    .add_local_file(Path(__file__).parent / "cfg_sudoku_att.yaml",
                    f"{TRM_REMOTE}/config/cfg_sudoku_att.yaml")
    .add_local_file(Path(__file__).parent.parent / "sudoku_rows.py", "/root/sudoku_rows.py")
)

app = modal.App("trm-eval")

# LDT volumes: the Sudoku cache (index-alignment check) and the eval root
# where per-puzzle rows are written.
ldt_data = modal.Volume.from_name("lattice-diffusion-data", create_if_missing=True)
ldt_ckpt = modal.Volume.from_name("lattice-diffusion-checkpoints", create_if_missing=True)

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
          out: str = "baselines/hrm_trm/results/trm_aug_sweep.json"):
    """Eval EVERY checkpoint in a run dir (exact+lenient) and save a JSON of
    {step, exact, lenient} for plotting accuracy-over-training."""
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


@app.function(image=image, gpu="B200", timeout=6 * 3600, memory=32768,
              volumes={"/ldt_data": ldt_data, "/ldt_ckpt": ldt_ckpt},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_sudoku(indices: list[int], variant: str = "mlp", checkpoint: str = "",
               batch_size: int = 1536, tag: str = "sudoku_test_perm") -> dict:
    """Exact-match accuracy on the given Sudoku-Extreme test indices, in order,
    via TRM's own run_eval_only.py (EMA weights, bf16 autocast).

    checkpoint="" -> the released alphaXiv checkpoint for `variant`. Writes
    per-puzzle rows + summary to the LDT checkpoint volume under
    eval/baselines/trm-<variant>/<tag>/."""
    import glob
    import subprocess
    import sys

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

    outdir = "/root/trm_sudoku_out"
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

    rows = sudoku_rows.build_rows(indices, labels, preds)
    summary = sudoku_rows.summarize(
        rows, model=f"TRM-{variant}", checkpoint=f"hf://{checkpoint}",
        trm_commit=TRM_COMMIT, config=config_name, index_list="sudoku_test_perm",
        gpu="B200", n_gpus=1, global_batch_size=batch_size,
        eval_seconds=eval_seconds, gpu_seconds=eval_seconds,
        index_aligned_with_ldt_cache=aligned,
    )
    sudoku_rows.write_outputs(f"/ldt_ckpt/eval/baselines/trm-{variant}/{tag}", rows, summary)
    ldt_ckpt.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def sudoku(n: int = 10_000, variant: str = "mlp", checkpoint: str = "",
           batch_size: int = 1536, out: str = ""):
    """Evaluate on the first n puzzles of sudoku_test_perm (LDT's ordering)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sudoku_rows

    indices = sudoku_rows.first_n_of_test_perm(n)
    summary = sudoku_rows.add_interval(run_sudoku.remote(
        indices, variant, checkpoint, batch_size, f"sudoku_test_perm_n{n}"))
    out = out or f"baselines/hrm_trm/results/trm_{variant}_sudoku.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}", flush=True)
