# E5 — Fine-tuned LLM baseline (Qwen3.5 on Sudoku-Extreme)

**Question.** Zero-shot frontier LLMs solve 0% of Sudoku-Extreme, but
zero-shot is a weak comparison for a task-trained solver: it leaves open
whether LDT's gains come from the recursive architecture + lattice
deduction, or simply from being *trained on the task*. A fine-tuned open
LLM on the identical 1K-puzzle training set isolates this.

**Design.**

- **Model:** `Qwen/Qwen3.5-0.8B`, fully fine-tuned in BF16 on one B200.
  The model is small enough that parameter-efficient tuning is unnecessary.
- **Supervision:** direct answer only: puzzle string → 81-character solution
  string. There are no generated search traces or trace-supervision variants.
- **Training data:** exactly 1,000 examples from the
  `sapientinc/sudoku-extreme` train split, selected with the LDT loader's
  fixed subset seed 42. No data augmentation is used in this first experiment.
- **Epoch sweep:** one run trains for five epochs by default and saves a
  resumable Hugging Face checkpoint after every epoch. Those checkpoints are
  the 1-, 2-, 3-, 4-, and 5-epoch conditions; `--epochs` changes the sweep
  ceiling.
- **Eval:** evaluate every epoch checkpoint on 32 held-out test puzzles
  selected with the LDT eval seed 200 by default. Draw 32 independent samples
  per puzzle and report the unbiased HumanEval pass@k curve for
  `k ∈ {1, 2, 4, 8, 16, 32}` from those samples. A completion is correct only
  if, after stripping outer whitespace, it is exactly 81 digits in `1`–`9` and
  equals the reference solution.
- **Artifacts:** use the deterministic Modal-volume directory
  `/checkpoints/followups/llm_baseline/qwen3_5_0_8b_seed<N>/`. Each
  `checkpoint-*` directory gets an `eval.json` and `eval.jsonl`; the run root
  gets `run_config.json` and `eval_all_epochs.json`.

The implementation is [`run.py`](run.py). It uses Transformers' native
Qwen3.5 support for both full SFT and generation, so training and evaluation
share the same tokenizer, chat template, and checkpoint format.

## Run

```bash
# Default: 5 epochs, 1,000 train puzzles, 32 eval puzzles, pass@32.
uv run modal run --detach followups/llm_baseline/run.py

# Short smoke test before spending the full evaluation budget.
uv run modal run followups/llm_baseline/run.py \
    --epochs 1 --n-train 32 --n-eval 2 --samples-per-puzzle 2
```

The default run is the experiment of record. The reduced command only checks
that model loading, checkpointing, and generation work end to end.

**Sanity gate.** Inspect the training loss and the one-epoch eval before
trusting the sweep. If no checkpoint can solve any training example when
sampled, debug the prompt/label masking rather than increasing the epoch
count.
