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
- **Epoch sweep:** one run trains for 16 epochs by default and saves a
  resumable Hugging Face checkpoint after every epoch. Evaluation runs only
  at epochs 2, 4, ..., 16 by default (`--eval-every-epochs 2`).
- **Hint / blank control:** `--n-blanks K` rebuilds every train and eval
  puzzle from its complete 81-digit solution, then chooses exactly `K` of the
  81 cell positions uniformly at random (without replacement) and replaces
  them with `0`. Thus the controlled blanks are **not** restricted to the
  original puzzle's blank positions: an original blank may be revealed, and
  an original given may be hidden. Selection is deterministic from the subset
  seed and puzzle index, but the sets are not nested across different `K`
  runs. Natural Sudoku-Extreme puzzles average ~56 blanks; omit the flag to
  preserve their original blank pattern (`.` normalized to `0`).
- **Eval:** evaluate every epoch checkpoint on 32 held-out test puzzles
  selected with the LDT eval seed 200 by default. Draw 32 independent samples
  per puzzle and report the unbiased HumanEval pass@k curve for
  `k ∈ {1, 2, 4, 8, 16, 32}` from those samples. A completion is correct only
  if, after stripping outer whitespace, it is exactly 81 digits in `1`–`9` and
  equals the reference solution.
- **Artifacts:** use the deterministic Modal-volume directory
  `/checkpoints/followups/llm_baseline/qwen3_5_0_8b_<blanksTag>_ep<E>_seed<N>/`,
  where `<blanksTag>` is `natural` or `blanksK`. Each `checkpoint-*`
  directory gets an `eval.json` and `eval.jsonl`; the run root gets
  `run_config.json` and `eval_all_epochs.json`.

The implementation is [`run.py`](run.py). It uses Transformers' native
Qwen3.5 support for both full SFT and generation, so training and evaluation
share the same tokenizer, chat template, and checkpoint format.

## Run

```bash
# Default: 16 epochs, eval at 2/4/.../16, natural blanks.
uv run modal run --detach followups/llm_baseline/run.py

# Controlled-blank sweep (same 16-epoch/even-epoch-eval regime).
uv run modal run --detach followups/llm_baseline/run.py --n-blanks 1
uv run modal run --detach followups/llm_baseline/run.py --n-blanks 2
uv run modal run --detach followups/llm_baseline/run.py --n-blanks 4
uv run modal run --detach followups/llm_baseline/run.py --n-blanks 8
uv run modal run --detach followups/llm_baseline/run.py --n-blanks 16
uv run modal run --detach followups/llm_baseline/run.py --n-blanks 32
```

The default run is the experiment of record.

**Sanity gate.** Inspect the training loss and the one-epoch eval before
trusting the sweep. If no checkpoint can solve any training example when
sampled, debug the prompt/label masking rather than increasing the epoch
count.

## Results

Seed-0 default run (natural ~56 blanks) recorded in [`results/`](results/)
and summarized in [`../RESULTS.md`](../RESULTS.md). Every epoch checkpoint
scored 0 on pass@1/2/4/8/16/32.

The controlled-blank sweep (`--n-blanks {1,2,4,8,16,32}`, 3 epochs) shows the
pipeline can learn: at 1–2 blanks, epoch-3 pass@32 reaches 100%; accuracy
falls as blanks increase and is still 0 at 32 blanks. See
[`results/blanks_sweep_summary.csv`](results/blanks_sweep_summary.csv).
