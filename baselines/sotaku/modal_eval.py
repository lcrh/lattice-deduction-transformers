"""Sotaku v2 on Sudoku-Extreme test puzzles in LDT's ordering, on one B200.

Runs the released v2.0.0 weights (commit below) with the release's inference
settings (FP32, batch 256, 1,024 iterations) on the first `n` puzzles of
`sudoku_test_perm`, and records exact-match accuracy and GPU seconds.

Usage:
    uv run modal run baselines/sotaku/modal_eval.py [--n 10000]
"""

import json
import os

import modal

SOTAKU_REMOTE = "/root/sotaku"
SOTAKU_COMMIT = "d0ab125"
WEIGHTS = "https://github.com/chenglou/sotaku/releases/download/v2.0.0/model_late_state_ce.pt"
RESULTS = os.path.join(os.path.dirname(__file__), "results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.2.6", "datasets==4.4.1", "pandas==2.3.3", "pyarrow==22.0.0",
                 "huggingface_hub==1.1.5")
    .run_commands(
        f"git clone https://github.com/chenglou/sotaku.git {SOTAKU_REMOTE}",
        f"cd {SOTAKU_REMOTE} && git checkout {SOTAKU_COMMIT}",
        f"curl -fsSL -o {SOTAKU_REMOTE}/model.pt {WEIGHTS}",
        f"curl -fsSL -o {SOTAKU_REMOTE}/model.pt.json {WEIGHTS}.json",
    )
)

app = modal.App("sotaku-eval")


@app.function(image=image, gpu="B200", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def evaluate(indices: list[int], iterations: int = 1024, batch_size: int = 256) -> dict:
    """Exact-match accuracy and GPU seconds on the given test-split indices."""
    import sys
    import time

    import torch
    sys.path.insert(0, SOTAKU_REMOTE)
    os.chdir(SOTAKU_REMOTE)
    from datasets import load_dataset
    from dataset_utils import DATASET_NAME, DATASET_REVISION
    from inference import RecurrentRunner
    from model_io import load_model
    from stabilize.exp_testbed_20k import encode_puzzles, encode_solutions

    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    model, _ = load_model("model.pt", device=device)
    test = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="test")
    rows = test.select(indices)
    inputs = encode_puzzles(rows["question"])
    targets = encode_solutions(rows["answer"]).long()
    empty = inputs[:, :, 0].bool()
    runner = RecurrentRunner(model)
    correct, seconds = 0, 0.0
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs, _ = runner.run_batch(inputs[start:end].to(device), [iterations],
                                      targets=targets[start:end].to(device),
                                      empty_mask=empty[start:end].to(device))
        pred = outputs[iterations].argmax(-1).cpu()
        torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        correct += int((pred == targets[start:end]).all(dim=1).sum())
    n = len(indices)
    return {"model": "Sotaku v2", "commit": SOTAKU_COMMIT, "n": n, "correct": correct,
            "wrong": n - correct, "exact_accuracy": correct / n, "iterations": iterations,
            "batch_size": batch_size, "gpu": "B200", "gpu_seconds": seconds,
            "index_list": "sudoku_test_perm"}


@app.local_entrypoint()
def main(n: int = 10000):
    from ldt.eval.indices import load

    indices = [int(i) for i in load("sudoku_test_perm").indices[:n]]
    result = evaluate.remote(indices)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "sotaku_v2_sudoku.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(result)
