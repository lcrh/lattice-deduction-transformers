# Follow-up experiments

Follow-on experiments for **Lattice Deduction Transformers**
([arXiv:2605.08605](https://arxiv.org/abs/2605.08605)). Each subdirectory is a
self-contained experiment with its own README describing the scientific
question, the run matrix, and the deliverable — always experimental data:
aggregated results (`results/`) and figures/tables (`plots/`).

The benchmark code these experiments build on lives in
[`../experiments/`](../experiments/) and [`../repro/`](../repro/)
(independent HRM/TRM reproductions).

> **Status: planning skeleton.** Every experiment README below is a design
> document with explicit `TODO(worker)` checklists. No follow-up code has
> been implemented yet unless a README says otherwise.

## Experiment index

| ID | Directory | Question | Domain | Est. cost |
|----|-----------|----------|--------|-----------|
| E1 | [`arch_ablation/`](arch_ablation/) | Which architectural / loss choices carry the result? (recursion & loop count incl. a data/compute escalation ladder, deep supervision, L_CE, soundness pressure) | Sudoku-Extreme | ~55 runs ≈ 14 B200-h |
| E2 | [`search_process/`](search_process/) | Search-process ablations: decision-selection heuristics (uniform vs. MRV vs. entropy-based vs. greedy) and backtracking policies (reset-to-root vs. partial/stochastic backjumping), matched between training and inference | Sudoku-Extreme | mostly eval-only + ~4 training configs |
| E3 | [`deduction_operator/`](deduction_operator/) | Ablating the deduction operator at inference: extra internal loops, deduce-to-fixpoint before branching, per-iteration elimination profiles, θ_elim sensitivity | Sudoku-Extreme | eval-only, reuses E1 checkpoints |
| E4 | [`ood_snowflake/`](ood_snowflake/) | Does LDT generalize to *unseen puzzle sizes*? Train on small Snowflake orders, test on larger ones | Snowflake | ~9 runs × ~5 B200-min |
| E5 | [`llm_baseline/`](llm_baseline/) | Does a *fine-tuned* LLM (Qwen3) close the gap, or is the architecture + lattice doing the work? | Sudoku-Extreme | a few GPU-hours (LoRA SFT) |
| E6 | [`latent_carry/`](latent_carry/) | Does carrying a TRM-style latent across solve steps help, or is the lattice a sufficient state? | Sudoku-Extreme | ~6 runs × ≤15 B200-min |
| E7 | [`maze_soundness/`](maze_soundness/) | Why does Maze-Hard emit valid-but-suboptimal paths instead of abstaining, and does a verification step restore soundness? | Maze-Hard | mostly eval/analysis |
| E8 | [`cost_accounting/`](cost_accounting/) | Normalized training-cost comparison (GPU-hours / FLOPs) + offline α-operator cost | all | bookkeeping + small benchmarks |

## Priority & ordering

**Tier 1 — start immediately, in parallel** (independent pipelines, no
contention):

1. **E1 `arch_ablation/`** — the fundamental experiment. It validates all
   the shared plumbing (config manifests, curve logging, collect/plot) and
   produces the checkpoints E2 and E3 consume. Nothing downstream starts
   cleanly until its D1 loop sweep exists.
2. **E4 `ood_snowflake/`** — the highest-leverage standalone result, and
   it runs on the Snowflake pipeline so it doesn't contend with E1 at all.
   Do the occupancy check and translation augmentation first (see its
   README — the positional confound is a blocker), then the transfer runs
   are cheap.

**Tier 2 — cheap, start as Tier 1 capacity frees up:**

3. **E8 `cost_accounting/`** — mostly bookkeeping on top of `repro/`
   measurements; fully independent; delivers the normalized cost table
   early.
4. **E2 `search_process/`** phase 1 — the eval-only policy scans run on
   existing benchmark checkpoints today; only the matched-training phase
   waits on S1 results.
5. **E3 `deduction_operator/`** — eval-only throughout. O1-scaling, O3
   fixpoint, and O4 thresholds run on the baseline checkpoint immediately;
   the transfer matrix and `d4_sym`/`d2_final_only` rows wait on E1.

**Tier 3 — independent but each needs its own new machinery:**

6. **E5 `llm_baseline/`** — separate SFT/eval stack; no dependency on the
   LDT runs, schedule opportunistically.
7. **E7 `maze_soundness/`** — forensics + verifier eval first; its
   training-side part is conditional on what the forensics show.

**Tier 4 — stretch:**

8. **E6 `latent_carry/`** — the most invasive change to shared model/
   trainer/solver code. Start only once E1–E3 have stabilized that code,
   so the carry plumbing doesn't churn under active ablation runs.

## Shared conventions

All experiments follow the conventions of the main repo unless their README
says otherwise:

- **Launch**: every run is a Modal app entrypoint, launched with
  `uv run modal run --detach <path> -- <flags>`. One launch command per run
  (no shell loops), so runs are individually addressable/cancellable.
- **Artifacts & checkpoint exchange — by deterministic volume path.**
  Training writes a checkpoint + `<ckpt>.eval.json` + `<ckpt>.eval.jsonl`
  to the `lattice-diffusion-checkpoints` Modal volume, exactly like
  `experiments/sudoku/run.py`, but at a **fixed, timestamp-free path**:

  ```
  /checkpoints/followups/<exp_id>/<config>_seed<N>.pt
  /checkpoints/followups/<exp_id>/<config>_seed<N>.eval.json[l]
  /checkpoints/followups/<exp_id>/<config>_seed<N>.train_curve.jsonl
  ```

  This path *is* the exchange contract: cross-experiment dependencies
  (e.g. E3 consuming E1's loop-sweep checkpoints) are expressed as these
  paths, never as ad-hoc handoffs — so worker agents run in isolation and
  find inputs where the READMEs say they are, and an independent
  reproducer who runs the same commands gets the identical layout on
  their own volume. Eval-only jobs write their outputs under the
  *consuming* experiment's directory, named
  `<evalconfig>__on__<input_config>_seed<N>.eval.json`. (The existing
  `train()` appends a wall-clock timestamp to checkpoint names — the
  followup runs need a name-override flag; see the E1 TODOs.)
- **Seeds**: 3 seeds (0, 1, 2) per config for anything that goes in a table;
  report mean and min–max range. 1 seed is fine for exploratory scans.
- **Baselines are re-run, not quoted**: every ablation table includes the
  unmodified baseline config re-run under the *same* eval protocol as the
  ablated rows, so numbers are internally comparable (previously published
  numbers used different eval sample sizes).
- **Results**: each experiment dir gets a `results/` folder with the
  collected `eval.json` summaries (small JSON/CSV committed to the repo) and
  a `plots/` output of its figure scripts, mirroring `repro/results/`.
- **Sanity gate**: before launching a sweep, re-run the baseline config once
  and check it reproduces the known number (Sudoku 4K-step: ~99.9–100% on the
  eval subsample). If it doesn't, stop and debug — don't burn the sweep.
