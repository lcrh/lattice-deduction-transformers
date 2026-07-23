# Follow-up experiment results

Snapshot: 2026-07-23. E1 has 81/81 evaluation summaries; E4 has 12/12.
All planned training runs have landed, plus nine paired same-weight D4
eval-only scans.

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

### Does LDT transfer learned constraint semantics to unseen puzzle sizes?

**Experiment.** Train Snowflake models on orders 4–5 or 4–6 and evaluate on
strictly larger orders. Translation augmentation addresses the absolute
position confound; a RoPE condition provides a relative-position control.
Models trained on all orders 4–8 form the in-distribution sanity control.

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

**Interpretation.** Local same-weight probing on `e4_leq5_seed0` /
`e4_leq6_seed0` / `e4_all_seed0` (MPS, 2026-07-23) shows the failure is
**not primarily a conflict-threshold artifact**. On OOD orders, the transfer
models' first deduction step already empties cells and kills ground-truth
candidates on ~83–90% of puzzles (CLS mean ≈ 1.0, empty-cell rate ≈ 83–90%,
unsound-elimination puzzles ≈ 83–90%). The in-distribution control stays clean
(CLS mean ≈ 0, empty/unsound ≈ 0). Disabling the CLS head
(`eval_cls_threshold=2.0`) only lifts short-horizon OOD accuracy from 0% to
~17–21%, still with zero wrong answers and mostly timeouts driven by empty-cell
resets. Making θ_elim more conservative down to 10⁻⁴ also does not help: on
OOD, ground-truth digit scores sit near 10⁻⁸–10⁻¹⁰, so they remain eliminated.
The weights do not transfer a sound deduction operator to larger orders.
