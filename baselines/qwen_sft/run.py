"""Fine-tuned LLM baseline: full BF16 SFT of Qwen3.5-0.8B on Sudoku-Extreme.

Supervision is direct answer only: puzzle string -> 81-digit solution string,
in the chat format of `make_messages`. Two puzzle setups:

- natural (default): the Sudoku-Extreme puzzle as given (~56 blanks);
- controlled blanks (`--n-blanks K`): each puzzle is rebuilt from its
  solution with exactly K uniformly chosen cells blanked. These cells are not
  restricted to the original blanks, and the sets are not nested across K.

Evaluation (the primary path; no training needed):
    uv run modal run baselines/qwen_sft/run.py::evaluate \
        --run-dir /checkpoints/followups/llm_baseline/qwen3_5_0_8b_natural_ep16_seed0

evaluates saved checkpoints on the first `n_eval` puzzles (default 256) of
LDT's `sudoku_test_perm` ordering, with `samples_per_puzzle` (default 32)
samples each, and reports unbiased pass@k. The puzzle setup (natural / K
blanks) is read from the run's `run_config.json`.

Training (optional; existing checkpoints are listed in baselines/README.md):
    uv run modal run --detach baselines/qwen_sft/run.py::train [--n-blanks K] [--seed S]

trains on a seed-42 draw of 1,000 puzzles from the whole train split (no
`sudoku_val` holdout, unlike LDT's pool) and evaluates every
`eval_every_epochs` epochs.

Needs the `llm` extra (transformers, accelerate, ...); the Modal image
installs it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal

from ldt.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT, HF_HOME, checkpoint_volume, data_volume, hf_secret,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
TRAIN_SUBSET_SEED = 42      # LDT's training subset seed
BLANK_SEED = 200            # seeds the controlled-blank cell choice per puzzle
SYSTEM_PROMPT = "You solve Sudoku puzzles. Follow the requested output format exactly."
USER_PROMPT = """Fill this Sudoku.

The puzzle is an 81-character row-major string. A 0 denotes a blank cell.
Return only the solved 81-digit string, with no spaces or explanation.

Puzzle: {question}"""
_SOLUTION_RE = re.compile(r"[1-9]{81}")
PASS_KS = (1, 2, 4, 8, 16, 32)


@dataclass
class TrainConfig:
    epochs: int = 16
    n_train: int = 1000
    n_blanks: int | None = None     # None = natural puzzles
    seed: int = 0
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 512
    eval_every_epochs: int = 2

    def run_name(self) -> str:
        return f"{blanks_tag(self.n_blanks)}_ep{self.epochs}_seed{self.seed}"


@dataclass
class EvalConfig:
    n_eval: int = 256               # first n of sudoku_test_perm
    samples_per_puzzle: int = 32
    batch_puzzles: int = 32         # x samples_per_puzzle sequences per generate call
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 0


def blanks_tag(n_blanks: int | None) -> str:
    return "natural" if n_blanks is None else f"blanks{n_blanks}"


def make_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    """The one prompt format shared by SFT and evaluation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=question)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def parse_solution(text: str) -> str | None:
    """Strictly parse a completion as one bare 81-digit Sudoku solution."""
    stripped = text.strip()
    return stripped if _SOLUTION_RE.fullmatch(stripped) else None


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c correct (HumanEval)."""
    if not (1 <= k <= n) or not (0 <= c <= n):
        raise ValueError(f"need 1 <= k <= n and 0 <= c <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_curve(n: int, c: int, ks: tuple[int, ...] = PASS_KS) -> dict[str, float]:
    return {f"pass_at_{k}": estimate_pass_at_k(n, c, k) for k in ks if k <= n}


def apply_n_blanks(answer: str, n_blanks: int, seed: int, puzzle_index: int) -> str:
    """The solution with exactly n_blanks cells replaced by '0'."""
    import numpy as np

    if not (0 <= n_blanks <= 81):
        raise ValueError(f"n_blanks must be in [0, 81], got {n_blanks}")
    if len(answer) != 81 or any(ch not in "123456789" for ch in answer):
        raise ValueError("answer must be an 81-digit Sudoku solution")
    cells = list(answer)
    rng = np.random.default_rng(seed + 1_000_003 * puzzle_index)
    for idx in rng.choice(81, size=n_blanks, replace=False):
        cells[int(idx)] = "0"
    return "".join(cells)


def make_rows(dataset, indices: list[int], n_blanks: int | None) -> list[dict[str, Any]]:
    """Prompt rows for the given dataset indices, in order."""
    rows = []
    for pos, idx in enumerate(indices):
        ex = dataset[int(idx)]
        question, answer = ex["question"], ex["answer"]
        if len(question) != 81 or len(answer) != 81:
            raise ValueError(f"malformed Sudoku-Extreme row {idx}")
        question = (question.replace(".", "0") if n_blanks is None
                    else apply_n_blanks(answer, n_blanks, BLANK_SEED, pos))
        rows.append({"idx": int(idx), "question": question, "answer": answer})
    return rows


def rows_digest(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{r['question']}\t{r['answer']}" for r in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def checkpoint_epoch(checkpoint: Path) -> float:
    state_path = checkpoint / "trainer_state.json"
    epoch = json.loads(state_path.read_text()).get("epoch")
    if epoch is None:
        raise ValueError(f"missing epoch in {state_path}")
    return float(epoch)


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("checkpoint-*"), key=checkpoint_epoch)


def select_checkpoints(checkpoints: list[Path], every_epochs: int) -> list[Path]:
    """Checkpoints whose rounded epoch is a positive multiple of `every_epochs`."""
    if every_epochs < 1:
        raise ValueError("every_epochs must be at least 1")
    out = []
    for checkpoint in checkpoints:
        epoch = int(round(checkpoint_epoch(checkpoint)))
        if epoch >= every_epochs and epoch % every_epochs == 0:
            out.append(checkpoint)
    return out


def summarize_records(records: list[dict[str, Any]], samples_per_puzzle: int) -> dict[str, Any]:
    n = len(records)
    return {
        "n_eval_puzzles": n,
        "samples_per_puzzle": samples_per_puzzle,
        **{f"pass_at_{k}": sum(r[f"pass_at_{k}"] for r in records) / n
           for k in PASS_KS if k <= samples_per_puzzle},
        "puzzles_passed": sum(r["n_correct"] > 0 for r in records),
        "correct_samples": sum(r["n_correct"] for r in records),
        "malformed_samples": sum(r["n_malformed"] for r in records),
        "total_samples": n * samples_per_puzzle,
    }


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------

# Qwen3.5's linear-attention layers need flash-linear-attention and causal-conv1d
# for fast kernels (transformers otherwise falls back to slow reference code).
# causal-conv1d is compiled against the locked torch (CUDA 13), hence the CUDA
# devel base image; only the B200 architecture is built.
llm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .uv_sync(extras=["llm"])
    .run_commands(
        "/.uv/.venv/bin/python -m ensurepip",
        "/.uv/.venv/bin/python -m pip install ninja packaging flash-linear-attention==0.5.2",
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=10.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 "
        "/.uv/.venv/bin/python -m pip install --no-build-isolation causal-conv1d==1.7.0",
    )
    .env({"HF_HOME": HF_HOME})
    .add_local_python_source("ldt")
)

app = modal.App("qwen-sudoku-sft")

RUNS_ROOT = f"{CHECKPOINT_MOUNT}/baselines/qwen_sft"
EVAL_ROOT = f"{CHECKPOINT_MOUNT}/eval/baselines/qwen_sft"
DEFAULT_RUN = f"{CHECKPOINT_MOUNT}/followups/llm_baseline/qwen3_5_0_8b_natural_ep16_seed0"


def _tokenizer(padding_side: str):
    """The base model's tokenizer (SFT does not change it), with a pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def _print_summaries(summaries: list[dict], samples_per_puzzle: int) -> None:
    for s in summaries:
        print(f"{Path(s['checkpoint']).name} epoch {s['epoch']:g}: " + "  ".join(
            f"pass@{k}={s[f'pass_at_{k}']:.3f}" for k in PASS_KS if k <= samples_per_puzzle))


def _load_dataset(split: str):
    from datasets import load_dataset

    return load_dataset("sapientinc/sudoku-extreme", split=split)


def _eval_rows(n_eval: int, n_blanks: int | None) -> list[dict[str, Any]]:
    from ldt.eval import indices

    test_idx = indices.load("sudoku_test_perm").indices[:n_eval].tolist()
    return make_rows(_load_dataset("test"), test_idx, n_blanks)


def _evaluate_checkpoint(checkpoint: Path, rows: list[dict[str, Any]], cfg: EvalConfig,
                         out_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText

    tokenizer = _tokenizer("left")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.config.use_cache = True
    model.eval()
    torch.manual_seed(cfg.seed)

    records = []
    t0 = time.time()
    n_batches = math.ceil(len(rows) / cfg.batch_puzzles)
    with torch.inference_mode():
        for start in range(0, len(rows), cfg.batch_puzzles):
            t_batch = time.time()
            batch = rows[start:start + cfg.batch_puzzles]
            prompts = [
                tokenizer.apply_chat_template(make_messages(r["question"]), tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
                for r in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda")
            outputs = model.generate(
                **inputs, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                num_return_sequences=cfg.samples_per_puzzle, max_new_tokens=cfg.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            completions = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for offset, row in enumerate(batch):
                texts = completions[offset * cfg.samples_per_puzzle:
                                    (offset + 1) * cfg.samples_per_puzzle]
                parsed = [parse_solution(t) for t in texts]
                n_correct = sum(p == row["answer"] for p in parsed)
                records.append({
                    "idx": row["idx"],
                    "pos": start + offset,
                    # records.py-compatible status: solved by any sample
                    # (pass@samples_per_puzzle); see n_correct for the rest.
                    "status": "correct" if n_correct else "wrong",
                    "question": row["question"],
                    "n_correct": n_correct,
                    "n_malformed": sum(p is None for p in parsed),
                    **pass_at_k_curve(cfg.samples_per_puzzle, n_correct),
                    "completions": texts,
                })
            n_samples = len(records) * cfg.samples_per_puzzle
            correct = sum(r["n_correct"] for r in records)
            print(f"  batch {start // cfg.batch_puzzles + 1}/{n_batches}: {len(records)} puzzles, "
                  f"pass@1 {correct / n_samples:.3f} ({correct}/{n_samples} samples), "
                  f"malformed {sum(r['n_malformed'] for r in records)}, "
                  f"{time.time() - t_batch:.1f}s", flush=True)
    summary = {
        "checkpoint": str(checkpoint),
        "epoch": checkpoint_epoch(checkpoint),
        "index_list": "sudoku_test_perm",
        "eval_digest_sha256": rows_digest(rows),
        "eval_config": asdict(cfg),
        "gpu": "B200",
        "eval_seconds": time.time() - t0,
        **summarize_records(records, cfg.samples_per_puzzle),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # Written last: a summary marks the checkpoint's eval as complete.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    del model
    torch.cuda.empty_cache()
    return summary


def _finished_summary(out_dir: Path, cfg: EvalConfig,
                      rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The saved summary of an identical, finished eval of this checkpoint, if any."""
    path = out_dir / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    same = (summary.get("eval_config") == asdict(cfg)
            and summary.get("eval_digest_sha256") == rows_digest(rows))
    return summary if same else None


@app.function(image=llm_image, gpu="B200", timeout=12 * 3600, secrets=[hf_secret],
              volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume})
def evaluate_run(run_dir: str, eval_cfg: dict, checkpoints: list[str] | None = None,
                 every_epochs: int = 2) -> list[dict[str, Any]]:
    """Evaluate the named checkpoint-* dirs of a run, or else every `every_epochs`-th epoch."""
    cfg = EvalConfig(**eval_cfg)
    run = Path(run_dir)
    run_config = json.loads((run / "run_config.json").read_text())
    n_blanks = run_config.get("n_blanks")
    if checkpoints:
        selected = [run / name for name in checkpoints]
    else:
        selected = select_checkpoints(checkpoint_dirs(run), every_epochs)
    if not selected:
        raise RuntimeError(f"no checkpoints selected in {run}")
    rows = _eval_rows(cfg.n_eval, n_blanks)
    tag = f"sudoku_test_perm_n{cfg.n_eval}_s{cfg.samples_per_puzzle}_seed{cfg.seed}"
    summaries = []
    for checkpoint in selected:
        out_dir = Path(EVAL_ROOT) / run.name / checkpoint.name / tag
        done = _finished_summary(out_dir, cfg, rows)
        if done is not None:
            print(f"already evaluated: {checkpoint}", flush=True)
            summaries.append({**done, "n_blanks": n_blanks})
            continue
        print(f"evaluating {checkpoint} ({blanks_tag(n_blanks)}) …", flush=True)
        summary = _evaluate_checkpoint(checkpoint, rows, cfg, out_dir)
        summary["n_blanks"] = n_blanks
        summaries.append(summary)
        checkpoint_volume.commit()
    return summaries


@app.function(image=llm_image, gpu="B200", timeout=12 * 3600, secrets=[hf_secret],
              volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume})
def train_run(train_cfg: dict, resume: bool = False) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import (
        AutoModelForImageTextToText, Trainer, TrainingArguments, set_seed,
    )

    from ldt.data.sudoku_extreme import SudokuExtremeConfig, select_pool

    cfg = TrainConfig(**train_cfg)
    set_seed(cfg.seed)
    run_dir = Path(RUNS_ROOT) / cfg.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = checkpoint_dirs(run_dir)
    if existing and not resume:
        raise FileExistsError(f"{run_dir} already has checkpoints; pass --resume")

    train_ds = _load_dataset("train")
    pool, _ = select_pool(len(train_ds), SudokuExtremeConfig(
        split="train", n_puzzles=cfg.n_train, seed=TRAIN_SUBSET_SEED, exclude_val=False))
    rows = make_rows(train_ds, np.asarray(pool).tolist(), cfg.n_blanks)
    (run_dir / "run_config.json").write_text(json.dumps({
        "model_id": MODEL_ID, **asdict(cfg),
        "train_subset_seed": TRAIN_SUBSET_SEED,
        "train_digest_sha256": rows_digest(rows),
    }, indent=2))
    checkpoint_volume.commit()

    tokenizer = _tokenizer("right")
    examples = []
    for r in rows:
        prompt = tokenizer.apply_chat_template(make_messages(r["question"]), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(make_messages(r["question"], r["answer"]),
                                             tokenize=False, add_generation_prompt=False,
                                             enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        enc = tokenizer(full, add_special_tokens=False, truncation=True, max_length=cfg.max_length)
        ids = enc["input_ids"]
        if len(ids) <= len(prompt_ids) or ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template changed or max_length truncates the answer")
        examples.append({"input_ids": ids, "attention_mask": enc["attention_mask"],
                         "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids):]})

    def collate(batch):
        def pad(key, value):
            return pad_sequence([torch.tensor(b[key]) for b in batch], batch_first=True,
                                padding_value=value)
        return {"input_ids": pad("input_ids", tokenizer.pad_token_id),
                "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run_dir), num_train_epochs=float(cfg.epochs),
            per_device_train_batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
            weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
            optim="adamw_torch_fused", bf16=True, tf32=True, gradient_checkpointing=True,
            logging_steps=1, save_strategy="epoch", save_total_limit=None,
            report_to="none", seed=cfg.seed, data_seed=cfg.seed, remove_unused_columns=False,
        ),
        train_dataset=examples,
        data_collator=collate,
        processing_class=tokenizer,
    )
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume and existing else None)
    train_seconds = time.time() - t0
    meta = json.loads((run_dir / "run_config.json").read_text())
    meta.update(train_seconds=train_seconds, gpu="B200")
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2))
    checkpoint_volume.commit()
    return {"run_dir": str(run_dir), "train_seconds": train_seconds}


@app.local_entrypoint()
def evaluate(run_dir: str = DEFAULT_RUN, checkpoint: str = "", every_epochs: int = 2,
             n_eval: int = 256, samples_per_puzzle: int = 32, seed: int = 0,
             batch_puzzles: int = 32, out: str = ""):
    """Evaluate an existing run's checkpoints (all even epochs, or one)."""
    cfg = EvalConfig(n_eval=n_eval, samples_per_puzzle=samples_per_puzzle, seed=seed,
                     batch_puzzles=batch_puzzles)
    summaries = evaluate_run.remote(run_dir, asdict(cfg),
                                    [checkpoint] if checkpoint else None, every_epochs)
    _print_summaries(summaries, samples_per_puzzle)
    if out:
        Path(out).write_text(json.dumps(summaries, indent=2))
        print(f"wrote {out}")


@app.local_entrypoint()
def train(epochs: int = 16, n_blanks: int | None = None, seed: int = 0,
          eval_every_epochs: int = 2, n_eval: int = 256, samples_per_puzzle: int = 32,
          resume: bool = False):
    """Train a new run, then evaluate it like `evaluate`."""
    cfg = TrainConfig(epochs=epochs, n_blanks=n_blanks, seed=seed,
                      eval_every_epochs=eval_every_epochs)
    result = train_run.remote(asdict(cfg), resume)
    print(json.dumps(result, indent=2))
    eval_cfg = EvalConfig(n_eval=n_eval, samples_per_puzzle=samples_per_puzzle, seed=seed)
    _print_summaries(evaluate_run.remote(result["run_dir"], asdict(eval_cfg), None,
                                         eval_every_epochs), samples_per_puzzle)
