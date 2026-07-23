# Follow-up experiments

Follow-on experiments for **Lattice Deduction Transformers**
([arXiv:2605.08605](https://arxiv.org/abs/2605.08605)). Each subdirectory is a
self-contained experiment with its own README describing the scientific
question, the run matrix, and the deliverable (a figure/table destined for a
paper appendix or a follow-up note).

The paper-reproduction code these experiments build on lives in
[`../experiments/`](../experiments/) (benchmarks) and [`../repro/`](../repro/)
(independent HRM/TRM reproductions).

> **Status: planning skeleton.** Every experiment README below is a design
> document with explicit `TODO(worker)` checklists. No follow-up code has
> been implemented yet unless a README says otherwise.

## Experiment index

| ID | Directory | Question | Domain | Est. cost |
|----|-----------|----------|--------|-----------|
| E1 | [`arch_ablation/`](arch_ablation/) | Which architectural / loss choices carry the result? (recursion & loop count, deep supervision, L_CE, soundness pressure) | Sudoku-Extreme | ~50 runs × ≤15 B200-min ≈ 10 B200-h |
| E2 | [`search_process/`](search_process/) | Search-process ablations: decision-selection heuristics (uniform vs. MRV vs. entropy-based vs. greedy) and backtracking policies (reset-to-root vs. partial/stochastic backjumping), matched between training and inference | Sudoku-Extreme | mostly eval-only + ~4 training configs |
| E3 | [`deduction_operator/`](deduction_operator/) | Ablating the deduction operator at inference: extra internal loops, deduce-to-fixpoint before branching, per-iteration elimination profiles, θ_elim sensitivity | Sudoku-Extreme | eval-only, reuses E1 checkpoints |
| E4 | [`ood_snowflake/`](ood_snowflake/) | Does LDT generalize to *unseen puzzle sizes*? Train on small Snowflake orders, test on larger ones | Snowflake | ~9 runs × ~5 B200-min |
| E5 | [`llm_baseline/`](llm_baseline/) | Does a *fine-tuned* LLM (Qwen3) close the gap, or is the architecture + lattice doing the work? | Sudoku-Extreme | a few GPU-hours (LoRA SFT) |
| E6 | [`latent_carry/`](latent_carry/) | Does carrying a TRM-style latent across solve steps help, or is the lattice a sufficient state? | Sudoku-Extreme | ~6 runs × ≤15 B200-min |
| E7 | [`maze_soundness/`](maze_soundness/) | Why does Maze-Hard emit valid-but-suboptimal paths instead of abstaining, and does a verification step restore soundness? | Maze-Hard | mostly eval/analysis |
| E8 | [`cost_accounting/`](cost_accounting/) | Normalized training-cost comparison (GPU-hours / FLOPs) + offline α-operator cost | all | bookkeeping + small benchmarks |

**Suggested order:** E1 first (it anchors the appendix and produces the
checkpoints E2/E3 reuse), then E2/E3/E8 (cheap, mostly eval-only), then E4
(OOD), then E5/E7, with E6 as the most invasive stretch item.

## Shared conventions

All experiments follow the conventions of the main repo unless their README
says otherwise:

- **Launch**: every run is a Modal app entrypoint, launched with
  `uv run modal run --detach <path> -- <flags>`. One launch command per run
  (no shell loops), so runs are individually addressable/cancellable.
- **Artifacts**: training writes a checkpoint + `<ckpt>.eval.json` +
  `<ckpt>.eval.jsonl` to the `lattice-diffusion-checkpoints` Modal volume,
  exactly like `experiments/sudoku/run.py`. Checkpoint names must embed the
  experiment ID and config name (e.g. `e1_loops8_cm_seed0_...`), so results
  can be collected by glob.
- **Seeds**: 3 seeds (0, 1, 2) per config for anything that goes in a table;
  report mean and min–max range. 1 seed is fine for exploratory scans.
- **Baselines are re-run, not quoted**: every ablation table includes the
  unmodified baseline config re-run under the *same* eval protocol as the
  ablated rows, so numbers are internally comparable (paper Table 1 numbers
  used a different eval sample size).
- **Results**: each experiment dir gets a `results/` folder with the
  collected `eval.json` summaries (small JSON/CSV committed to the repo) and
  a `plots/` output of its figure scripts, mirroring `repro/results/`.
- **Sanity gate**: before launching a sweep, re-run the baseline config once
  and check it reproduces the known number (Sudoku 4K-step: ~99.9–100% on the
  eval subsample). If it doesn't, stop and debug — don't burn the sweep.
