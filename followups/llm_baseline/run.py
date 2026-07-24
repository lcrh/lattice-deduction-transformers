"""Full-finetune Qwen3.5-0.8B on Sudoku-Extreme and eval every epoch.

Usage:
    uv run modal run --detach followups/llm_baseline/run.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import modal

from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT,
    DATA_MOUNT,
    checkpoint_volume,
    data_volume,
    hf_secret,
    image,
)


MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_RUN_NAME = "qwen3_5_0_8b"
SYSTEM_PROMPT = (
    "You solve Sudoku puzzles. Follow the requested output format exactly."
)
USER_PROMPT = """Fill this Sudoku.

The puzzle is an 81-character row-major string. A 0 denotes a blank cell.
Return only the solved 81-digit string, with no spaces or explanation.

Puzzle: {question}"""
_SOLUTION_RE = re.compile(r"[1-9]{81}")


def make_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    """Build the one prompt format shared by SFT and evaluation."""
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
    if k < 1:
        raise ValueError("k must be at least 1")
    if n < 1:
        raise ValueError("n must be at least 1")
    if c < 0 or c > n:
        raise ValueError("c must be between 0 and n inclusive")
    if k > n:
        raise ValueError(f"k={k} exceeds the number of samples n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_curve(
    n: int,
    c: int,
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict[str, float]:
    """Estimate pass@k for each requested k that fits in n samples."""
    return {f"pass_at_{k}": estimate_pass_at_k(n, c, k) for k in ks if k <= n}


def _dataset_digest(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(f"{row['question']}\t{row['answer']}" for row in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _load_rows(split: str, n: int, seed: int) -> list[dict[str, str]]:
    import numpy as np
    from datasets import load_dataset

    dataset = load_dataset("sapientinc/sudoku-extreme", split=split)
    if n > len(dataset):
        raise ValueError(f"Requested {n} {split} examples, but split has {len(dataset)}")
    # Match SudokuExtremeDataset's subset selection exactly so E5 sees the
    # same 1K train subset (seed 42) and held-out ordering (seed 200) as LDT.
    indices = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    dataset = dataset.select(sorted(indices.tolist()))
    rows = [{"question": row["question"], "answer": row["answer"]} for row in dataset]
    for row in rows:
        if len(row["question"]) != 81 or len(row["answer"]) != 81:
            raise ValueError(f"Malformed {split} row in sapientinc/sudoku-extreme")
    return rows


def _tokenize_training_rows(
    rows: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, list[int]]]:
    tokenized: list[dict[str, list[int]]] = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            make_messages(row["question"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            make_messages(row["question"], row["answer"]),
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = full["input_ids"]
        if len(input_ids) <= len(prompt_ids):
            raise ValueError("max_length truncates the entire assistant answer")
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Qwen chat template changed: training prompt is not a prefix")
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": full["attention_mask"],
                "labels": labels,
            }
        )
    return tokenized


def _checkpoint_epoch(checkpoint: Path) -> float:
    state_path = checkpoint / "trainer_state.json"
    with state_path.open() as fh:
        state = json.load(fh)
    if state.get("epoch") is None:
        raise ValueError(f"Missing epoch in {state_path}")
    return float(state["epoch"])


def _checkpoint_dirs(run_dir: Path) -> list[Path]:
    checkpoints = list(run_dir.glob("checkpoint-*"))
    return sorted(checkpoints, key=_checkpoint_epoch)


app = modal.App("qwen35-sudoku-finetune")


@app.function(
    image=image,
    gpu="B200",
    timeout=3600 * 12,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def train_and_evaluate(
    epochs: int = 5,
    n_train: int = 1000,
    n_eval: int = 32,
    samples_per_puzzle: int = 32,
    train_batch_size: int = 32,
    eval_batch_puzzles: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    import os

    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if min(n_train, n_eval, samples_per_puzzle, train_batch_size, eval_batch_puzzles) < 1:
        raise ValueError("dataset sizes, sample counts, and batch sizes must be positive")

    os.environ["HF_HOME"] = f"{DATA_MOUNT}/huggingface"
    set_seed(seed)
    run_dir = Path(
        f"{CHECKPOINT_MOUNT}/followups/llm_baseline/{DEFAULT_RUN_NAME}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = _checkpoint_dirs(run_dir)
    if existing and not resume:
        raise FileExistsError(
            f"{run_dir} already has checkpoints; pass --resume or choose another seed"
        )

    print("Loading deterministic train/test subsets...", flush=True)
    train_rows = _load_rows("train", n_train, 42)
    eval_rows = _load_rows("test", n_eval, 200)
    run_config = {
        "model_id": MODEL_ID,
        "epochs": epochs,
        "n_train": n_train,
        "n_eval": n_eval,
        "samples_per_puzzle": samples_per_puzzle,
        "train_batch_size": train_batch_size,
        "eval_batch_puzzles": eval_batch_puzzles,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "train_subset_seed": 42,
        "eval_subset_seed": 200,
        "train_digest_sha256": _dataset_digest(train_rows),
        "eval_digest_sha256": _dataset_digest(eval_rows),
    }
    with (run_dir / "run_config.json").open("w") as fh:
        json.dump(run_config, fh, indent=2)
    checkpoint_volume.commit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    training_rows = _tokenize_training_rows(train_rows, tokenizer, max_length)

    class TokenizedDataset(Dataset):
        def __len__(self) -> int:
            return len(training_rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return training_rows[index]

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [torch.tensor(row["input_ids"]) for row in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [torch.tensor(row["attention_mask"]) for row in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [torch.tensor(row["labels"]) for row in batch],
                batch_first=True,
                padding_value=-100,
            ),
        }

    print(f"Loading {MODEL_ID} for full BF16 fine-tuning...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
        save_only_model=False,
        save_total_limit=None,
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(),
        data_collator=collate,
        processing_class=tokenizer,
    )

    train_started = time.time()
    trainer.train(resume_from_checkpoint=True if resume and existing else None)
    train_seconds = time.time() - train_started
    checkpoint_volume.commit()

    del trainer, model, training_rows
    torch.cuda.empty_cache()
    tokenizer.padding_side = "left"

    def evaluate_checkpoint(checkpoint: Path) -> dict[str, Any]:
        epoch = _checkpoint_epoch(checkpoint)
        print(f"Evaluating {checkpoint.name} (epoch {epoch:g})...", flush=True)
        eval_model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        eval_model.config.use_cache = True
        eval_model.eval()
        torch.manual_seed(seed + int(round(epoch * 1000)))

        puzzle_records: list[dict[str, Any]] = []
        total_correct_samples = 0
        total_malformed = 0
        with torch.inference_mode():
            for start in range(0, len(eval_rows), eval_batch_puzzles):
                batch_rows = eval_rows[start : start + eval_batch_puzzles]
                prompt_texts = [
                    tokenizer.apply_chat_template(
                        make_messages(row["question"]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for row in batch_rows
                ]
                inputs = tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to("cuda")
                outputs = eval_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=samples_per_puzzle,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completions = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                for offset, row in enumerate(batch_rows):
                    lo = offset * samples_per_puzzle
                    texts = completions[lo : lo + samples_per_puzzle]
                    parsed = [parse_solution(text) for text in texts]
                    correct = [solution == row["answer"] for solution in parsed]
                    n_correct = sum(correct)
                    total_correct_samples += n_correct
                    total_malformed += sum(solution is None for solution in parsed)
                    puzzle_records.append(
                        {
                            "kind": "puzzle",
                            "puzzle_index": start + offset,
                            "question": row["question"],
                            "answer": row["answer"],
                            "passed": n_correct > 0,
                            "n_correct_samples": n_correct,
                            "n_malformed_samples": sum(
                                solution is None for solution in parsed
                            ),
                            **pass_at_k_curve(samples_per_puzzle, n_correct),
                            "completions": texts,
                        }
                    )

        n_passed = sum(record["passed"] for record in puzzle_records)
        n_samples = len(eval_rows) * samples_per_puzzle
        pass_metrics = {
            f"pass_at_{k}": sum(record[f"pass_at_{k}"] for record in puzzle_records)
            / len(puzzle_records)
            for k in (1, 2, 4, 8, 16, 32)
            if k <= samples_per_puzzle
        }
        summary = {
            "checkpoint": str(checkpoint),
            "epoch": epoch,
            "n_eval_puzzles": len(eval_rows),
            "samples_per_puzzle": samples_per_puzzle,
            **pass_metrics,
            "puzzles_passed": n_passed,
            "correct_samples": total_correct_samples,
            "malformed_samples": total_malformed,
            "total_samples": n_samples,
            "generation": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
        }
        with (checkpoint / "eval.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
        with (checkpoint / "eval.jsonl").open("w") as fh:
            fh.write(json.dumps({"kind": "header", **summary}) + "\n")
            for record in puzzle_records:
                fh.write(json.dumps(record) + "\n")

        del eval_model
        torch.cuda.empty_cache()
        checkpoint_volume.commit()
        return summary

    checkpoints = _checkpoint_dirs(run_dir)
    if len(checkpoints) < epochs:
        raise RuntimeError(
            f"Expected at least {epochs} epoch checkpoints, found {len(checkpoints)}"
        )
    summaries = [evaluate_checkpoint(checkpoint) for checkpoint in checkpoints]
    all_epochs = {
        "run_config": run_config,
        "train_seconds": train_seconds,
        "epochs": summaries,
    }
    with (run_dir / "eval_all_epochs.json").open("w") as fh:
        json.dump(all_epochs, fh, indent=2)
    checkpoint_volume.commit()
    return all_epochs


@app.local_entrypoint()
def entrypoint(
    epochs: int = 5,
    n_train: int = 1000,
    n_eval: int = 32,
    samples_per_puzzle: int = 32,
    train_batch_size: int = 32,
    eval_batch_puzzles: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    resume: bool = False,
) -> None:
    result = train_and_evaluate.remote(
        epochs=epochs,
        n_train=n_train,
        n_eval=n_eval,
        samples_per_puzzle=samples_per_puzzle,
        train_batch_size=train_batch_size,
        eval_batch_puzzles=eval_batch_puzzles,
        learning_rate=learning_rate,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        resume=resume,
    )
    print(json.dumps(result, indent=2), flush=True)
