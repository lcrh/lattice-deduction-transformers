# E5 — Fine-tuned LLM baseline (Qwen3 on Sudoku-Extreme)

**Question.** The paper compares against *zero-shot* frontier LLMs (which
solve 0%). That leaves open whether LDT's gains come from the recursive
architecture + lattice deduction, or simply from being *trained on the
task*. A fine-tuned open LLM on the identical 1K-puzzle training set
isolates this.

**Design.**

- **Models:** Qwen3-4B and Qwen3-8B (instruct variants), LoRA fine-tuned.
  One model is enough to make the point; two sizes show the trend.
- **Data parity:** the same 1K Sudoku-Extreme training puzzles LDT uses.
  Two supervision formats, both worth running because they bracket the
  fairness spectrum:
  1. **Direct answer**: puzzle string → 81-char solution string. Matches
     LDT's information diet exactly (puzzle + solution, nothing else).
  2. **Search-trace CoT**: puzzle → serialized solver trace (e.g. from
     `experiments/sudoku/dpll.py`-style propagation + branching by a
     symbolic solver) → solution. More favorable to the LLM; if even this
     fails, the point is made strongly.
- **Augmentation parity:** run both with and without the digit-perm ×
  dihedral augmentation LDT uses (as data expansion for the SFT set).
- **Eval:** same 1,000-puzzle test subsample as E1, greedy decoding plus
  best-of-N sampling at N matched to LDT's inference-compute per puzzle
  (report both). Strict 81-cell exact match; also report cells-correct so
  a near-miss profile is visible.
- **Report:** params, training GPU-hours, and inference cost next to
  accuracy — the table row should slot directly into the paper's Table 1
  format (this experiment also feeds E8's normalized-cost table).

**Expected outcome (to be falsified):** low-but-nonzero direct-answer
accuracy, better-but-still-far CoT accuracy at ~4 orders of magnitude more
parameters and compute than LDT's 800K/15-min budget. Whatever the number,
it replaces the weakest comparison in the paper with a meaningful one.

**Cost.** LoRA SFT on 1K–8K examples is minutes-to-an-hour per config on a
single A100/H100-class GPU; eval dominated by best-of-N sampling. Budget a
few GPU-hours total across {2 models × 2 formats × ±aug}. Trim the grid to
{4B × both formats, 8B × best format} if needed.

## TODO(worker)

- [ ] Decide serving/training stack on Modal (e.g. HF `peft` + `trl` SFT
      for training; vLLM for batched eval) and pin it in this README.
- [ ] `make_sft_data.py`: emit JSONL for both formats from the
      `sapientinc/sudoku-extreme` train split (reuse
      `lattice_diffusion/data/sudoku_extreme.py`); trace generator for
      format 2 (a plain symbolic DPLL with a serialization — do NOT use the
      learned model; the trace must be solver-ground-truth).
- [ ] `train_sft.py` (Modal app): LoRA config, both model sizes, logs to
      W&B like `repro/` does.
- [ ] `eval_llm.py`: batched generation, strict parser (reject malformed
      grids as wrong, count separately), best-of-N with a
      validity-then-majority pick; writes `eval.json` in the same schema
      family as the LDT evals so `collect.py` patterns transfer.
- [ ] Prompt formats documented in a `prompts.md` (the paper appendix
      already has zero-shot prompts to stay consistent with).
- [ ] Sanity gates: (1) zero-shot Qwen3 baseline first (expect ~0%,
      consistent with the paper's frontier-LLM result); (2) SFT model must
      reach ~100% on *training* puzzles, otherwise the pipeline (not the
      model) is the bottleneck.
