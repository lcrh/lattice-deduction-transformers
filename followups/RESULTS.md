# Follow-up experiment results

Snapshot: 2026-07-23. E1 has 77/81 evaluation summaries; E4 has 12/12.
Together, 89/93 planned runs have landed.

Pass rates below are means of the available per-seed percentages. Most
evaluations use 1,000 puzzles per seed. Low-performing reruns can stop after
50 timeouts and report the maximal contiguous evaluated prefix, so their
denominators can be smaller. Wrong answers and timeouts remain separate:
a timeout is an abstention, while a wrong answer is a soundness failure.

## E1 — Architecture and loss ablations

### D1-C1: Is recurrence necessary at a fixed parameter budget?

**Experiment.** Hold the tied 4-layer, approximately 800K-parameter backbone
fixed and vary the number of recurrent loops.

**Results.**

| loops | mean pass rate |
|---:|---:|
| 1 | 17.1% |
| 2 | 41.6% |
| 4 | 91.9% |
| 8 | 98.8% |
| 16 | 99.1% |
| 32 | 99.0% |

The 16-loop baseline reaches 99.2% across its separate control runs. Most
C1 failures are timeouts rather than wrong answers.

**Interpretation.** At fixed parameters and optimizer steps, recurrence has a
large effect. Performance rises sharply through eight loops and then reaches
the ceiling. A one- or two-pass transformer does not make effective use of the
same outer search process under this budget.

### D1-C2: Can additional training compute or data buy back recurrence?

**Experiment.** Increase training steps for low-loop models while keeping the
architecture fixed. The final two L=1 conditions use four times the baseline
training compute, with one also using the full training split.

**Results.**

- L=2 at training-FLOPs parity: 95.1%.
- L=1 at training-FLOPs parity: 44.7%.
- `d1_L1_cm4x`, two seeds: in progress.
- `d1_L1_bigdata`, two seeds: in progress.

**Interpretation.** Additional compute substantially helps, especially at
L=2, but the completed L=1 parity run remains far below the recurrent
baseline. The decisive test of whether four-times compute or unrestricted data
can close the L=1 gap is still in progress, so the stronger capability-gap
claim is not yet supported.

### D1-C3: Can a static parameter-matched shape replace recurrence?

**Experiment.** Compare non-recurrent models with roughly 800K parameters,
allocating those parameters across different depth/width shapes.

**Results.**

- 4×128 (`d1_L1`): 17.1%.
- 8×92: 29.8%.
- 16×64: 6.0%.
- 32×44: 4.3%.
- Tied 4×128×16-loop baseline: 99.2%.

**Interpretation.** No tested static allocation of the same parameter budget
comes close to the recurrent model. Moderate extra depth helps relative to the
one-pass 4-layer model, but making the static model narrower and deeper
eventually hurts.

### D1-C4: Can more untied parameters replace recurrence?

**Experiment.** Give non-recurrent models more parameters and enough steps to
match or exceed the recurrent baseline's training compute.

**Results.**

- Untied 8-layer, approximately 1.6M parameters: 85.8%.
- Untied 16-layer, approximately 3.2M parameters: 96.7%.
- Wide 4×256, approximately 3.2M parameters: 31.2%.
- Untied 16-layer maximum-budget condition, approximately 3.2M parameters,
  four-times compute, and the full split: 99.85% across two seeds
  (999/1000 and 998/1000, zero wrong answers).

**Interpretation.** Recurrence is not strictly required for ceiling
performance: the no-excuses untied model matches the baseline. It does so with
roughly four times the parameters, four times the training compute, and more
data. The result therefore supports a strong parameter/data/compute-efficiency
advantage for weight-tied recurrence, not an absolute capability separation.
The poor wide-model result also shows that parameter count alone is not enough;
how the extra capacity is allocated matters.

### D2: Does deep supervision matter?

**Experiment.** Compare supervision at all 16 internal iterations with
supervision only at the final iteration, keeping inference final-only.

**Results.**

- All-iteration baseline: 99.2%.
- Final-only supervision: 96.1%.

**Interpretation.** Deep supervision improves the 2K-step endpoint by about
3.1 percentage points, but final-only supervision still performs strongly.
The stated question is partly about learning speed and iteration transfer, so
the endpoint alone is not decisive. The training curves and E3 loop-transfer
evaluation still need to be analyzed.

### D3: Does the auxiliary per-cell cross-entropy help?

**Experiment.** Vary `softmax_loss_weight` while leaving the rest of the
baseline fixed.

**Results.**

- λ_ce=0: 24.4%.
- λ_ce=0.2 baseline: 99.2%.
- λ_ce=1: 99.93%.

The λ_ce=0 evaluations early-aborted at roughly 72–77 puzzles per seed after
reaching the timeout limit; all three had zero wrong answers.

**Interpretation.** The auxiliary cross-entropy is essential for learning
under the 2K-step budget, and increasing its weight from 0.2 to 1 reaches the
ceiling. These endpoint results do not yet distinguish whether the gain comes
from better deductions or from the softmax head's branching policy; calls per
solve, unsound-elimination diagnostics, and learning curves should determine
the channel.

### D4: What enforces soundness?

**Experiment.** Vary BCE asymmetry and remove the dedicated CLS conflict head.
The original runs all used the baseline deduction threshold θ_elim=0.1.

**Results at fixed θ_elim=0.1.**

- Symmetric BCE, ratio 1×: 24.7%.
- BCE ratio 2×: 50.3%.
- Baseline BCE ratio 8×: 99.2%.
- BCE ratio 32×: 97.8%.
- Ratio 8× without the CLS conflict head: 21.2%.

The no-CLS condition returns wrong answers on 2,365/3,000 puzzles, rather than
mostly timing out. By contrast, the low-asymmetry conditions with the conflict
head mostly abstain.

**Adjusted-threshold companion results.**

- Independently trained symmetric models evaluated at θ_elim=0.5: 18.0%.
- Independently trained ratio-2 models evaluated at θ_elim=0.333: 23.5%.
- Independently trained ratio-32 models evaluated at θ_elim=0.0303: 98.9%.

**Paired same-weight eval-only results.** The original fixed-0.1 checkpoints
were re-evaluated without retraining:

- Original symmetric checkpoints at θ_elim=0.5: 17.6%.
- Original ratio-2 checkpoints at θ_elim=0.33: 24.7%.
- Original ratio-32 checkpoints at θ_elim=0.03: 96.2%.

**Interpretation.** The CLS conflict head is crucial to the empirical
soundness behavior: removing it turns abstentions into confidently wrong
answers. Greater BCE asymmetry also correlates with higher solve rate. The
paired eval isolates inference calibration: ratio-matched θ_elim does not
rescue the low-asymmetry checkpoints and slightly reduces every row relative
to fixed θ_elim=0.1 (24.7% → 17.6% for symmetric, 50.3% → 24.7% for ratio-2,
and 97.8% → 96.2% for ratio-32). The independently trained companions agree
qualitatively. The original ordering is therefore not an artifact of using
θ_elim=0.1; stronger BCE asymmetry genuinely produced a much more useful
deduction operator under the tested operating points.

## E4 — Snowflake out-of-distribution order transfer

### Does LDT transfer learned constraint semantics to unseen puzzle sizes?

**Experiment.** Train Snowflake models on orders 4–5 or 4–6 and evaluate on
strictly larger orders. Translation augmentation addresses the absolute
position confound; a RoPE condition provides a relative-position control.
Models trained on all orders 4–8 form the in-distribution sanity control.

**Results.**

- Trained and evaluated on orders 4–8: 600/600 correct, zero wrong answers,
  and zero timeouts. Every individual order reaches 100%.
- Trained on orders 4–5 and evaluated on 6–8: 0/600 correct, zero wrong
  answers, and 600 timeouts.
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
resets. So the weights do not transfer a sound deduction operator to larger
orders: conflict-threshold retuning may recover a small fraction of puzzles,
but it cannot repair the underlying unsound eliminations. A CLS scan remains
worth running as a secondary check; it should not be expected to reverse the
headline 0% result.
