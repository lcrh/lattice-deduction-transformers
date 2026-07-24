# Follow-up experiment results

Snapshot: 2026-07-23. E1 has 81/81 evaluation summaries. E4 has the original
transfer runs plus the sparse-support grid (H × {1K,4K} × {abs,RoPE}, 1 seed
per cell; some 4K-abs cells also have 3 seeds) and 8K absolute probes at
H∈{12,25,50}. E5 has the first Qwen3.5-0.8B seed-0 epoch sweep.

Pass rates below are means of the per-seed percentages. Most evaluations use
1,000 puzzles per seed. Low-performing reruns can stop after 50 timeouts and
report the maximal contiguous evaluated prefix, so their denominators can be
smaller. Wrong answers and timeouts remain separate: a timeout is an
abstention, while a wrong answer is a soundness failure.

**Train data.** Unless noted otherwise, E1 trains on a fixed 1,000-puzzle
subsample of `sapientinc/sudoku-extreme` (`--n-train-puzzles 1000`). “Full
split” means that flag is unset, so training draws from the entire HF train
split (~3.83M puzzles). That removes the 1K-data bottleneck; it does not
change the eval set.

**Inference cost on solved examples.** `calls/solve` is
`model_calls_total / correct` (outer search model invocations per correct
puzzle). `avg rounds` is `avg_rounds_solved` (mean DPLL rounds among puzzles
that solved). Both are means across seeds. Low-accuracy rows are noisier:
they often early-abort, and the solved set is a small, easy subset.

## E1 — Architecture and loss ablations

### D1-C1: Is recurrence necessary at a fixed parameter budget?

**Experiment.** Hold the tied 4-layer, approximately 800K-parameter backbone
fixed and vary the number of recurrent loops. This sweep is **not**
compute-matched: fewer loops mean fewer FLOPs per training step.

**Results.**

| loops | mean pass rate | calls/solve | avg rounds |
|---:|---:|---:|---:|
| 1 | 17.1% | 665 | 303 |
| 2 | 41.6% | 204 | 218 |
| 4 | 91.9% | 34 | 178 |
| 8 | 98.8% | 12 | 83 |
| 16 | 99.1% | 9.5 | 62 |
| 32 | 99.0% | 10 | 67 |

The 16-loop baseline reaches 99.2% across its separate control runs
(10 calls/solve, 70 avg rounds). Most C1 failures are timeouts rather than
wrong answers. As loops increase, solved puzzles also become cheaper at
inference: fewer outer calls and fewer DPLL rounds.

**Interpretation.** At fixed parameters and optimizer steps, recurrence has a
large effect on both accuracy and inference efficiency. Performance rises
sharply through eight loops and then reaches the ceiling. Because training
FLOPs scale with loops, this alone does not separate “needs recursion” from
“undertrained fewer-loop models”; that is D1-C2.

### D1-C2: Can additional training compute or data buy back recurrence?

**Experiment.** Increase training steps for low-loop models so they are closer
to (or beyond) the recurrent baseline’s training FLOPs, while keeping the
architecture fixed. The final two L=1 conditions use four times the baseline
training compute; `d1_L1_bigdata` also switches from the default 1K subsample
to the full ~3.83M-puzzle train split, so longer training is not limited to
replaying the same 1K puzzles.

**Results.**

| condition | mean pass rate | calls/solve | avg rounds |
|---|---:|---:|---:|
| L=2 FLOPs-parity (`d1_L2_cm`) | 95.1% | 26 | 151 |
| L=1 FLOPs-parity (`d1_L1_cm`) | 44.7% | 190 | 237 |
| L=1 4× compute (`d1_L1_cm4x`, 2 seeds) | 62.2% | 97 | 153 |
| L=1 4× + full train split (`d1_L1_bigdata`, 2 seeds) | 57.4% | 114 | 164 |
| L=16 baseline | 99.2% | 10 | 70 |

**Interpretation.** The C1 loop sweep understates what fewer loops can do once
training compute is matched. L=2 at FLOPs parity nearly recovers the recurrent
ceiling (95.1%, with inference still ~2–3× more expensive than baseline). L=1
improves with compute (17% → 45% at parity → ~62% at 4×) but plateaus well
below baseline, and unrestricted data does not help further (57.4%).

So recursion is necessary under the budgets tested here: a single pass does not
reach the recurrent solve rate even with substantial extra training. Opening the
full train split on top of 4× compute does not help further (57.4% vs 62.2% on
the 1K subsample), so the remaining L=1 gap is not explained by exhausting a
tiny puzzle pool. Fewer recursion steps may still suffice with additional
training — L=2 already almost does — but buying back all the way to L=1 has not
worked yet.

### D1-C3: Can a static parameter-matched shape replace recurrence?

**Experiment.** Compare non-recurrent models with roughly 800K parameters,
allocating those parameters across different depth/width shapes.

**Results.**

| shape | mean pass rate | calls/solve | avg rounds |
|---|---:|---:|---:|
| 4×128 (`d1_L1`) | 17.1% | 665 | 303 |
| 8×92 | 29.8% | 330 | 253 |
| 16×64 | 6.0% | 2935 | 196 |
| 32×44 | 4.3% | 15340 | 316 |
| Tied 4×128×16-loop baseline | 99.2% | 10 | 70 |

**Interpretation.** No tested static allocation of the same parameter budget
comes close to the recurrent model. Moderate extra depth helps relative to the
one-pass 4-layer model, but making the static model narrower and deeper
eventually hurts. The rare solves are also extremely expensive in outer calls.

### D1-C4: Can more untied parameters replace recurrence?

**Experiment.** Give non-recurrent models more parameters and enough steps to
match or exceed the recurrent baseline's training compute.

**Results.**

| condition | mean pass rate | calls/solve | avg rounds |
|---|---:|---:|---:|
| Untied 8-layer (~1.6M) | 85.8% | 46 | 206 |
| Untied 16-layer (~3.2M) | 96.7% | 21 | 129 |
| Wide 4×256 (~3.2M) | 31.2% | 310 | 229 |
| Untied 16-layer max (~3.2M, 4× compute, full train split; 2 seeds) | 99.85% | 4.4 | 32 |
| Tied 16-loop baseline | 99.2% | 10 | 70 |

**Interpretation.** Recurrence is not strictly required for ceiling
performance: the no-excuses untied model matches the baseline. It does so with
roughly four times the parameters, four times the training compute, and more
data — and those solves are actually cheaper at inference (4.4 calls/solve vs
10). The result therefore supports a strong parameter/data/compute-efficiency
advantage for weight-tied recurrence, not an absolute capability separation.
The poor wide-model result also shows that parameter count alone is not enough;
how the extra capacity is allocated matters.

### D2: Does deep supervision matter?

**Experiment.** Compare supervision at all 16 internal iterations with
supervision only at the final iteration, keeping inference final-only.

**Results.**

| condition | mean pass rate | calls/solve | avg rounds |
|---|---:|---:|---:|
| All-iteration baseline | 99.2% | 10 | 70 |
| Final-only supervision | 96.1% | 22 | 133 |

**Interpretation.** Deep supervision improves the 2K-step endpoint by about
3.1 percentage points, and the all-iteration model also solves with roughly
half the outer search cost. Final-only supervision still performs strongly.
The stated question is partly about learning speed and iteration transfer, so
the endpoint alone is not decisive. The training curves and E3 loop-transfer
evaluation still need to be analyzed.

### D3: Does the auxiliary per-cell cross-entropy help?

**Experiment.** Vary `softmax_loss_weight` while leaving the rest of the
baseline fixed.

**Results.**

| λ_ce | mean pass rate | calls/solve | avg rounds |
|---:|---:|---:|---:|
| 0 | 24.4% | 424 | 75 |
| 0.2 (baseline) | 99.2% | 10 | 70 |
| 1 | 99.93% | 1.8 | 12 |

The λ_ce=0 evaluations early-aborted at roughly 72–77 puzzles per seed after
reaching the timeout limit; all three had zero wrong answers.

**Interpretation.** The auxiliary cross-entropy is essential for learning
under the 2K-step budget. Raising its weight from 0.2 to 1 reaches the
ceiling and also yields the cheapest solves in the whole E1 suite
(~2 calls/solve). These endpoint results do not yet distinguish whether the
gain comes from better deductions or from the softmax head's branching
policy; unsound-elimination diagnostics and learning curves should determine
the channel.

### D4: What enforces soundness?

**Experiment.** Vary BCE asymmetry and remove the dedicated CLS conflict head.
The original runs all used the baseline deduction threshold θ_elim=0.1.

**Results at fixed θ_elim=0.1.**

| condition | mean pass rate | calls/solve | avg rounds | notes |
|---|---:|---:|---:|---|
| Symmetric BCE (1×) | 24.7% | 389 | 8 | mostly timeouts |
| BCE ratio 2× | 50.3% | 157 | 210 | mostly timeouts |
| Baseline BCE 8× | 99.2% | 10 | 70 | |
| BCE ratio 32× | 97.8% | 15 | 93 | |
| Ratio 8×, no CLS head | 21.2% | 27 | 44 | 2,365/3,000 wrong |

The no-CLS condition returns wrong answers rather than mostly timing out. By
contrast, the low-asymmetry conditions with the conflict head mostly abstain.
(The very low `avg rounds` on `d4_sym` reflects that the few solves finish
quickly; most puzzles never solve.)

**Adjusted-threshold companion results.** Independently trained models with
matched θ_elim=1/(1+ratio):

| condition | mean pass rate | calls/solve | avg rounds |
|---|---:|---:|---:|
| Symmetric @ θ=0.5 | 18.0% | 581 | 35 |
| Ratio-2 @ θ=0.33 | 23.5% | 414 | 12 |
| Ratio-32 @ θ=0.03 | 98.9% | 10 | 69 |

**Paired same-weight eval-only results.** The original fixed-0.1 checkpoints
re-evaluated without retraining:

| condition | mean pass rate | calls/solve |
|---|---:|---:|
| Symmetric @ θ=0.5 | 17.6% | 573 |
| Ratio-2 @ θ=0.33 | 24.7% | 418 |
| Ratio-32 @ θ=0.03 | 96.2% | 21 |

**Interpretation.** The CLS conflict head is crucial to the empirical
soundness behavior: removing it turns abstentions into confidently wrong
answers. Greater BCE asymmetry also correlates with higher solve rate and
cheaper solves. The paired eval isolates inference calibration: ratio-matched
θ_elim does not rescue the low-asymmetry checkpoints and reduces every row
relative to fixed θ_elim=0.1 (24.7% → 17.6% for symmetric, 50.3% → 24.7% for
ratio-2, and 97.8% → 96.2% for ratio-32). The independently trained companions
agree qualitatively. The original ordering is therefore not an artifact of
using θ_elim=0.1; stronger BCE asymmetry genuinely produced a much more useful
deduction operator under the tested operating points.

## E4 — Snowflake out-of-distribution order transfer

### Does LDT transfer learned constraint semantics across order shifts?

**Experiment.** Train Snowflake models on orders 4–5 or 4–6 and evaluate on
strictly larger orders. Translation augmentation addresses the absolute
position confound; a RoPE condition provides a relative-position control.
Models trained on all orders 4–8 form the in-distribution sanity control.
A support-preserving distribution-shift condition (`e4_shift95`) also trains
on all orders, but allocates 475/500 training puzzles to orders 4–5 and only
25/500 to orders 6–8 (exact counts: 238, 237, 9, 8, 8). A follow-on
sparse-support grid then fixes 1K total puzzles and sweeps higher-order count
H, training steps, and absolute vs RoPE position encodings, with 8K absolute
probes at H∈{12,25,50}.

**Results.**

- Trained and evaluated on orders 4–8: 600/600 correct, zero wrong answers,
  and zero timeouts. Every individual order reaches 100%. Solved puzzles take
  **1.0 call/solve** on average (essentially one-shot deductions).
- Trained on orders 4–5 and evaluated on 6–8: 0/600 correct, zero wrong
  answers, and 600 timeouts (no solved examples, so no calls/solve).
- Trained on orders 4–6 and evaluated on 7–8: 0/600 correct, zero wrong
  answers, and 600 timeouts.
- Trained on orders 4–5 with RoPE and evaluated on 6–8: 0/600 correct, zero
  wrong answers, and 600 timeouts.
- Trained on the 95% lower-order mixture (500 puzzles, 4K steps) and evaluated
  evenly across orders 4–8: **587/600 correct (97.8%)**, 13 wrong answers, and
  zero timeouts (1.02 calls/solve). Per-order pass rates are 100%, 100%,
  96.6%, 96.7%, and 96.8% for orders 4 through 8, respectively. On under-
  represented orders 6–8 alone: **380/393 (96.7%)**.

**Sparse-support grid.** Hold total training puzzles at 1K and vary (a) higher-
order count H ∈ {3,6,12,25,50} (examples of orders 6–8 combined), (b) steps ∈
{1000,4000}, and (c) position encoding ∈ {absolute, RoPE}. Metric focus is
accuracy on orders **6–8** (n=131 per seed). Cells are 1 seed unless noted.

| H | 1K abs | 1K RoPE | 4K abs | 4K RoPE |
|---:|---:|---:|---:|---:|
| 3 | 26% | 19% | 33% (3 seeds) | 43% |
| 6 | 34% | 36% | 43% (3 seeds) | 56% |
| 12 | 51% | 70% | 77% (3 seeds) | 87% |
| 25 | 71% | 86% | 88% (3 seeds) | **100%** |
| 50 | 82% | 92% | 93% (3 seeds) | 97% |

**8K absolute probes** (compute vs sparse support):

| H | overall | orders 6–8 |
|---:|---:|---:|
| 12 | 94% (188/200) | 91% (119/131) |
| 25 | 99% (198/200) | 98.5% (129/131) |
| 50 | 99.8% (599/600, 3 seeds) | 99.7% (392/393) |

**Interpretation.** Local same-weight probing on `e4_leq5_seed0` /
`e4_leq6_seed0` / `e4_all_seed0` (MPS, 2026-07-23) shows the strict-omission
failure is **not primarily a conflict-threshold artifact**. On OOD orders, the
transfer models' first deduction step already empties cells and kills ground-
truth candidates on ~83–90% of puzzles (CLS mean ≈ 1.0, empty-cell rate ≈
83–90%, unsound-elimination puzzles ≈ 83–90%). The in-distribution control stays
clean (CLS mean ≈ 0, empty/unsound ≈ 0). Disabling the CLS head
(`eval_cls_threshold=2.0`) only lifts short-horizon OOD accuracy from 0% to
~17–21%, still with zero wrong answers and mostly timeouts driven by empty-cell
resets. Making θ_elim more conservative down to 10⁻⁴ also does not help: on
OOD, ground-truth digit scores sit near 10⁻⁸–10⁻¹⁰, so they remain eliminated.
The weights do not transfer a sound deduction operator to completely unseen
larger orders.

The soft-shift results sharply qualify that failure. Giving orders 6–8 only
25 total training examples (the original `e4_shift95`) raises higher-order
performance from 0% under strict omission to 96.7%. The 1K-puzzle grid then
shows how far that can be pushed:

- **H is the main lever.** H≤6 remains weak (~19–56% on 6–8). The cliff is
  around H≈12; H≥25 approaches the balanced `e4_all` ceiling.
- **Training steps matter at mid H.** At H=12, going 1K→4K lifts 6–8 accuracy
  from 51% to 77% (abs); 8K reaches 91%. Paper-budget 1K steps are not enough
  for sparse support.
- **RoPE helps once support is non-trivial**, especially at short budgets
  (H=12 @1K: 51%→70%; H=25 @1K: 71%→86%). It does not rescue H=3. At H=25 with
  4K steps, RoPE reaches 100% on 6–8; at H=50, RoPE@4K (97%) nearly matches
  abs@8K (99.7%).
- Residual errors under soft shift are almost entirely **wrong answers**, not
  timeouts — a small soundness cost versus the balanced control's zero wrongs.

Thus the catastrophic transfer failure is mainly an **out-of-support
extrapolation failure**, not fragility to a large change in order frequencies.
Tens of higher-order examples plus enough optimization steps recover nearly all
accuracy; a handful of examples do not.

## E5 — Fine-tuned Qwen3.5-0.8B baseline

### Does task-trained direct-answer SFT close the Sudoku-Extreme gap?

**Experiment.** Fully fine-tune `Qwen/Qwen3.5-0.8B` in BF16 on the fixed
1,000-puzzle LDT train subset (subset seed 42), with no augmentation and no
search-trace supervision. Save a checkpoint after every epoch for five epochs
(~3 minutes total train on one B200). Evaluate each checkpoint on 32 held-out
test puzzles (subset seed 200) with 32 samples per puzzle and the unbiased
HumanEval pass@k curve for `k ∈ {1, 2, 4, 8, 16, 32}`. Artifacts live under
`followups/llm_baseline/results/` and on the Modal volume at
`/checkpoints/followups/llm_baseline/qwen3_5_0_8b_seed0/`.

**Results (seed 0).**

| epoch | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | malformed / 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0% | 0% | 0% | 0% | 0% | 0% | 344 |
| 2 | 0% | 0% | 0% | 0% | 0% | 0% | 178 |
| 3 | 0% | 0% | 0% | 0% | 0% | 0% | 249 |
| 4 | 0% | 0% | 0% | 0% | 0% | 0% | 275 |
| 5 | 0% | 0% | 0% | 0% | 0% | 0% | 270 |

At epoch 5, 754/1024 samples are well-formed 81-digit strings with digits
`1`–`9`, but none match the reference solution. Wrong well-formed grids
average only ~15/81 cells correct (max 31), so the model mostly learns the
output format rather than the puzzle.

**Interpretation.** Under this direct-answer SFT setup, five epochs of
Qwen3.5-0.8B on the same 1K train set do not produce any correct held-out
Sudoku-Extreme solutions even at pass@32. That turns the zero-shot LLM
comparison into a stronger negative: task training alone, without lattice
search, is not enough for this model/data budget on natural ~56-blank
puzzles.

### Controlled blank sweep (sanity + difficulty ladder)

**Experiment.** Same model/data/eval protocol, but rebuild every train and
eval puzzle from its solution with exactly `K ∈ {1, 2, 4, 8, 16, 32}` blanks.
Train for 3 epochs. This checks that the SFT/eval pipeline can learn anything
at all, then measures where performance collapses.

**Results (seed 0, epoch 3).**

| blanks | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80.9% | 90.3% | 95.8% | 98.7% | 99.9% | 100% |
| 2 | 95.1% | 97.8% | 99.4% | 99.9% | 100% | 100% |
| 4 | 73.7% | 82.8% | 88.4% | 90.3% | 90.6% | 90.6% |
| 8 | 37.6% | 47.9% | 58.7% | 69.2% | 77.2% | 81.2% |
| 16 | 5.7% | 9.9% | 16.1% | 24.3% | 34.3% | 46.9% |
| 32 | 0% | 0% | 0% | 0% | 0% | 0% |

**Interpretation.** The pipeline is not broken: with 1–2 missing cells the
model reaches perfect pass@32 by epoch 3. Performance degrades smoothly with
blank count and is already weak at 16 blanks; at 32 blanks (still far easier
than natural Sudoku-Extreme) it returns to 0%, matching the natural-blank
failure mode.
