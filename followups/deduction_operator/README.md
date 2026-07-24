# E3 — Ablating the deduction operator at inference

> Internal planning doc — will be tightened up before this dir goes
> public-facing.

**Question.** At inference the learned deduction operator runs at a fixed
operating point: 16 internal loops, one deduction pass per solve round,
θ_elim = 0.1, θ_CLS = 0.6. All of these are *inference-time* choices — the
weights are loop-count-invariant and the thresholds are post-hoc — so a
frozen checkpoint can be probed across the whole operating range without
retraining. How much deduction is the model actually capable of per
forward, and can extra inference compute (more loops, iterated deduction)
substitute for training compute?

**Testbed.** Entirely **eval-only**. Primary checkpoint: E1 baseline (2K
steps). The loop-transfer matrix additionally consumes E1's D1-C1 sweep
(`L_train ∈ {1,2,4,8,16,32}`) and the `d2_final_only` checkpoint; the
θ-sweep on symmetric BCE consumes E1's `d4_sym`. **Blocked on E1's
training runs** — no coordination beyond the filesystem: E1 checkpoints
appear at deterministic Modal-volume paths
(`/checkpoints/followups/e1/<config>_seed<N>.pt`, per the conventions in
`followups/README.md`), and this experiment's evals start as those paths
materialize.

---

## Sub-study O1 — Inference loop scaling & train/eval loop transfer

If we let the model think longer per forward than it was trained to —
more internal loops — does it deduce more? And does a model trained at
one depth work when unrolled to another?

The backbone is a weight-tied loop, so a checkpoint trained at `L_train`
can be *evaluated* at any `L_eval` (the model docstring notes stability to
128+ loops with improving NLL).

- **Scaling on the baseline:** `L_eval ∈ {1, 2, 4, 8, 16, 32}`,
  full solve-loop eval. Does more looping raise solve rate / reduce
  branching, and where does it saturate? Since wallclock per forward
  scales with `L_eval`, also report the *compute-honest* view: solve rate
  vs. total forward-FLOPs per puzzle, so "more loops" competes fairly
  against "more search rounds".
- **Transfer matrix:** `L_train ∈ {1, 2, 4, 8, 16, 32}` × `L_eval ∈ {1, 2,
  4, 8, 16, 32}` on the 200-puzzle subsample (36 cells × 2 seeds
  = 72 eval jobs). Hypothesis: deep supervision trains every iteration to be a
  valid readout, so transfer should be broad — and should *break* on the
  `d2_final_only` checkpoint (6 cells × 2 seeds = 12 extra eval jobs), which ties this study back
  to E1-D2.

**Figure O1a**: solve rate + unsound rate vs `L_eval` (log-x) for the
baseline, with a FLOPs-normalized companion panel. **Figure O1b**: heatmap
of solve rate over (L_train, L_eval), plus the `d2_final_only` row.

## Sub-study O2 — Per-iteration elimination profile

Inside a single forward, what does each successive iteration actually
contribute — and where does it stop contributing?

One `return_all=True` forward exposes all intermediate iterations' heads —
so the "what does each extra loop buy?" question can be answered *per
forward*, cleanly separated from search effects.

On a fixed **state bank** (states harvested from baseline solve
trajectories, stratified by fill level — early/mid/late-solve), run a
single forward at `L_eval = 128` and measure per iteration ℓ:

- candidates that would be eliminated at θ_elim (cumulative and marginal),
- how many of those eliminations are unsound (GT bit killed — the bank
  keeps each state's ground truth),
- CLS logit trajectory on SAT vs. UNSAT states (does conflict confidence
  grow monotonically in ℓ?).

**Figure O2**: eliminations/forward and unsound rate vs ℓ, one line per
fill-level stratum; CLS trajectory panel. This is the mechanistic
companion to O1's end-to-end curves.

## Sub-study O3 — Deduce-to-fixpoint before branching

Should the solver keep deducing until nothing more falls out before it
risks a guess — instead of guessing after a single pass?

Orthogonal to internal loop count: the outer solve loop currently runs
**one** deduction pass per round, then immediately branches if the state
is still undecided. Alternative: iterate (forward → threshold-eliminate)
on the same state until no candidate falls below threshold — a fixpoint in
the lattice — and only then decide. Classic unit-propagation-to-fixpoint,
with a learned propagator.

| config | deduction passes per round |
|---|---|
| baseline | 1 |
| `o3_d2`, `o3_d4` | capped at 2, 4 |
| `o3_fix` | until fixpoint (safety cap ~16) |

Things this changes, worth measuring separately:

- **Branching pressure**: decisions per solve should drop — each decision
  is the risky step, so trading forwards for decisions may raise both
  solve rate and soundness.
- **Unsound compounding**: repeated application of an approximate operator
  can compound errors — each pass eliminates at θ_elim again on an already-
  tightened state. Track unsound rate per pass index.
- **Augmentation ensembling**: each `dpll_step` call wraps the forward in a
  fresh random symmetry, so iterated deduction is also an implicit
  test-time ensemble over augmentations. To separate the two effects, run
  `o3_d4` with eval-time aug disabled as a control.
- **Honest cost accounting**: `model_calls` per round is now > 1; report
  forwards/solve, not rounds/solve.

If fixpoint deduction wins clearly, a *matched-training* variant (trainer
also deduces to fixpoint between gradient steps, changing the pool's state
distribution) is the follow-on — that run belongs methodologically with
E2's matched/mismatched machinery; note it there rather than duplicating.

## Sub-study O4 — Operating-point sensitivity (θ_elim, θ_CLS)

Are the two inference thresholds sitting on comfortable plateaus, or on
knife-edges we happened to land on?

- θ_elim ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.5} on the baseline checkpoint:
  is the asymmetry-matched 0.1 on a plateau or a knife-edge? Report solve
  rate, unsound rate, wrong answers, calls/solve.
- The same sweep on E1's `d4_sym` checkpoint (symmetric BCE): the best
  point on that curve is the "symmetric + post-hoc tuned threshold"
  comparison — testing whether asymmetric-BCE-with-matched-threshold
  actually beats it.
- θ_CLS^eval ∈ {0.5, 0.55, 0.6, 0.7, 0.8} on the baseline: sensitivity of
  the false-restart / missed-conflict trade-off around the tuned 0.6.

**Figure O4**: three sensitivity curves (solve rate + wrong answers vs
threshold), baseline and `d4_sym` overlaid on the θ_elim panel.

---

## Deliverables

Figures O1a/O1b, O2, O4 + a table for O3 (passes-per-round vs solve rate /
forwards-per-solve / decisions-per-solve / unsound rate / wrong answers).
Together they answer one question from four angles: how much deduction
does one forward buy, and what do more buy?

## Run budget

All eval-only: ~48 transfer evals + ~30 scaling/threshold/fixpoint evals
× ~1–4 B200-min ≈ **3–4 B200-hours**, plus the O2 profiler (~minutes).

## TODO(worker) — implementation checklist

- [ ] Eval-time loop override: `--eval-n-loops N` in `run.py` and
      `eval_only.py` (rebuild `LoopedTransformerConfig` with the override
      before `load_state_dict`; weights are loop-invariant). Sanity: a
      baseline checkpoint evaluated at its native L must reproduce its
      number exactly.
- [ ] `StepConfig.deduce_passes: int = 1` (0 = fixpoint mode with safety
      cap) — implement in `dpll_step` as an inner loop around forward +
      threshold-eliminate, deciding only after the last pass. Conflict
      check (empty-cell / CLS) after every pass, early-exit on conflict or
      all-singleton. Keep default behavior byte-identical.
- [ ] Per-pass diagnostics: extend `info` with per-pass deduce counts +
      unsound counts (when GT available) so O3's compounding analysis
      falls out of the standard eval jsonl.
- [ ] `state_bank.py` — harvest + save the stratified state bank (state,
      GT, fill level, SAT/UNSAT label) from baseline solve trajectories;
      commit the generation script, store the bank on the Modal volume.
- [ ] `profile_iters.py` — O2: single `return_all=True` forward at L=128
      over the bank; emit per-iteration CSV.
- [ ] `configs.py` / `collect.py` / `plot_all.py` mirroring
      `arch_ablation/` conventions; eval-only launch commands target
      `eval_only.py --checkpoint …`.
- [ ] Consume E1 checkpoints strictly by their fixed volume paths
      (`/checkpoints/followups/e1/{d1_L*,d2_final_only,d4_sym}_seed<N>.pt`);
      don't retrain anything here. Write this experiment's eval outputs
      under `/checkpoints/followups/e3/` with the
      `<evalconfig>__on__<input>_seed<N>` naming from the conventions.
- [ ] Sanity gates: (1) `--eval-n-loops 16` == no-flag baseline; (2)
      `deduce_passes=1` == current behavior; (3) O2 profiler's iteration-16
      elimination counts consistent with what the end-to-end eval sees.
