# E1 — Architecture & loss ablations (Sudoku-Extreme)

**Question.** The headline LDT config bundles several design choices —
a weight-tied recurrent backbone unrolled 16×, per-iteration ("deep")
supervision, an auxiliary per-cell cross-entropy `L_CE`, and asymmetric
soundness pressure in the BCE + a dedicated conflict head. Which of these
actually carry the result, and by how much?

Three specific claims/choices this experiment puts numbers on:

- **Recursion**: is a weight-tied recurrent backbone necessary at all,
  given the lattice scaffolding already makes the *outer* process
  incremental? A single-pass transformer inside the same search loop is
  the natural null hypothesis.
- **`L_CE` speeds up learning**: a working assumption with no supporting
  result so far.
- **Soundness pressure**: the asymmetric BCE and CLS conflict head are the
  mechanism behind the empirical-soundness claim; ablating them should
  show wrong answers reappearing.

**Testbed.** Sudoku-Extreme, 1K-puzzle train split. Baseline = the
standard config trained for **2,000 steps** (800K params, ~7 min on 1×
B200). Deliberately *not* the 4K-step headline budget: at 4K the model
sits at the ~100% ceiling where ablation deltas vanish; at 2K it is just
below ceiling (~99% accuracy, still zero wrong answers), so degradations
are visible — and every run halves in cost. Re-check any contested or
surprising row at 4K before drawing conclusions about the headline
setting. Every ablation is a one-factor deviation from this baseline —
no factorial crossing.

**Deliverable.** The main ablation table (`results/summary.csv`, one row
per config × seed with mean/range roll-ups) plus three figures, described
sub-study by sub-study below. Inference-time behavior of the deduction
operator (extra loops at eval, per-iteration elimination profiles, θ_elim
sensitivity) is deliberately **not** here — that is
[E3 `deduction_operator/`](../deduction_operator/), which reuses this
experiment's checkpoints.

---

## Baseline

```bash
uv run modal run --detach experiments/sudoku/run.py \
    --steps 2000 --n-train-puzzles 1000 --n-eval-puzzles 1000 --seed 0
```

Baseline hyperparameters (from `experiments/sudoku/run.py` defaults):
4 layers × 16 loops, dim 128, batch 512, `bce_pos_mult/neg_mult = 4.0/0.5`
(asymmetry ratio 8×), `softmax_loss_weight = 0.2` (`L_CE`),
`conflict_loss_weight = 0.1`, θ_elim 0.1, all-iteration supervision,
final-iteration readout at inference (`use_final=True` in `dpll_step`).

**Eval protocol for the whole experiment:** fixed 1,000-puzzle test
subsample (`--n-eval-puzzles 1000`, dataset seed 200), 3 training seeds per
config. The full-split eval (`eval_only.py`) is reserved for rows
that end up near ceiling and need extra resolution. Primary metrics, all
already emitted in `<ckpt>.eval.json`:

- **accuracy** (`correct/n`) and **soundness** (`wrong` count — a wrong
  *returned* answer, distinct from a timeout),
- **search effort**: model calls per solved puzzle (p50/p90 from the
  per-puzzle jsonl),
- **unsound-elimination rate** (`diag.unsound_rate`) — fraction of
  eliminations that killed a ground-truth candidate,
- **conflict-head precision/recall** (`diag.conflict_*`),
- **train wallclock** (`train_wallclock.post_compile_secs`).

---

## Sub-study D1 — Is recursion necessary? (loops vs. depth vs. params)

Three quantities trade off when you ablate the recurrence: **parameters**,
**per-forward compute** (layer-applications per forward), and **training
compute** (steps × per-forward). Weight tying is exactly the mechanism
that decouples the first two — the tied 4-layer × L-loop model holds
params at 800K while per-forward compute scales with L. There is *no*
non-recurrent architecture that matches the baseline on both at once
(untying multiplies params; shrinking width to compensate changes the
model class) — that impossibility is part of the finding, and the grid
below is built to surface it: every comparison names what it holds fixed.

**C1 — Tied loop sweep (params fixed at 800K; primary).** `L = n_loops ∈
{1, 2, 4, 8, 16, 32}` at the fixed 4-layer, dim-128 backbone, all at
`steps = 2000` (identical optimizer steps, pool dynamics, and data —
per-forward compute is the only thing that varies; train wallclock scales
≈ 7 min × L/16). `L = 1` is the non-recurrent null hypothesis: a plain
4-layer transformer doing one-shot lattice elimination inside the same
outer search.

**C2 — Can data/compute buy back the recursion? (escalation ladder).**
The obvious objection to C1's low-L results: "L=1 got 16× less training
compute — train it longer." The working hypothesis is stronger than
parity: the non-looped model stays far behind *even with more data and
compute than the baseline ever saw* — a capability gap, not a budget gap.
Test it as a ladder on `L = 1` (plus one `L = 2` parity point):

| config | n_loops | steps | train puzzles | training FLOPs vs baseline |
|---|---|---|---|---|
| `d1_L2_cm` | 2 | 16,000 | 1K | 1× (parity) |
| `d1_L1_cm` | 1 | 32,000 | 1K | 1× (parity) |
| `d1_L1_cm4x` | 1 | 128,000 | 1K | 4× |
| `d1_L1_bigdata` | 1 | 128,000 | full split (`--n-train-puzzles` unset) | 4×, data bottleneck removed |

(Warmup/cosine schedule are step-fractions so they rescale automatically;
`max_age` stays 100 pool steps.) The decisive evidence is the
`train_curve.jsonl` trajectory, not just the endpoint: a curve that has
**plateaued** below baseline at 4× compute with unrestricted data is a
capability gap; a curve **still climbing** means budget could eventually
close it and the claim must be stated as a compute-efficiency gap
instead. The 4× rungs cost ~30 B200-min each — run them at 2 seeds.

**C3 — Depth without tying (per-forward compute matched, params grow).**
Non-recurrent (`L = 1`) with more *untied* layers, compared against the
tied config with the same layer-applications per forward:

| config | layers × loops | dim | fwd layer-apps | params |
|---|---|---|---|---|
| tied `d1_L2` / `d1_L4` (from C1) | 4 × {2, 4} | 128 | 8 / 16 | 800K |
| `d1_untied8` | 8 × 1 | 128 | 8 | ~1.6M |
| `d1_untied16` | 16 × 1 | 128 | 16 | ~3.2M |

If tied wins *with 2–4× fewer params*, iteration + input re-injection
beats plain depth outright. If untied wins, C4 decides whether that's
depth or just capacity.

**C4 — Capacity control (params matched to untied, depth is not the
mechanism).** One wide non-recurrent run: `d1_wide` = 4 layers × L=1,
dim 256 → ~3.2M params, ≈16 dim-128-layer-equivalents of forward compute.
This completes a triangle at roughly equal params and forward FLOPs:
`d1_untied16` (deep) vs `d1_wide` (wide) vs tied `d1_L4` (iterated, at
4× fewer params) — separating iteration, depth, and capacity.

All configs at `steps = 2000` except C2. ~12 configs + baseline; 3 seeds
except the C2 4× rungs (2 seeds).

**Figure D1** (`plots/d1_loops.pdf`): solve rate (second panel:
unsound-elimination rate) vs per-forward layer-applications, log-x —
C1 as the main line, C3/C4 as scatter points (marker size ∝ params,
shape = tied/untied/wide), C2 as annotated open markers on the L=1/L=2
positions. 3 seeds, mean ± range. Companion panel
(`plots/d1_escalation_curves.pdf`): C2 learning curves (solve count vs
training FLOPs, log-x) with the baseline curve overlaid — the
plateaued-vs-still-climbing evidence.

The C1 checkpoints double as E3's `L_train` axis; E3 reads them from
their fixed volume paths
(`/checkpoints/followups/e1/d1_L<L>_seed<N>.pt`).

## Sub-study D2 — Deep supervision (Sotaku-style supervise-every-iteration)

The baseline supervises all 16 internal iterations and averages the loss
(`_losses` in `experiments/sudoku/train.py`). Ablation: supervise the
**final iteration only**, keeping inference readout unchanged (it is
already final-only, so the comparison is clean).

| config | supervision |
|---|---|
| `baseline` | all 16 iterations |
| `d2_final_only` | final iteration only |

The interesting readout is the **learning curve**, not just the endpoint:
the in-train mini-solve (every 100 steps, max_rounds 5) counts puzzles
solved by pure deduction — a direct learning-speed proxy. A second
hypothesis — that deep supervision is what makes the *per-iteration
refinement* monotone, and hence what makes loop-count transfer at
inference work — is tested in E3 by running its profile/transfer evals on
the `d2_final_only` checkpoint.

**Figure D2** (`plots/d2_curves.pdf`): in-train solve count vs step,
baseline vs final-only, 3 seeds each.

## Sub-study D3 — Does `L_CE` speed up learning?

Working assumption under test: adding `L_CE` makes the model learn
faster. Measure it with learning curves:

| config | softmax_loss_weight |
|---|---|
| `d3_ce0` | 0.0 |
| `baseline` | 0.2 |
| `d3_ce1` (optional) | 1.0 |

**Figure D3** (`plots/d3_ce_curves.pdf`): in-train solve count vs step +
accuracy/soundness at 500/1K/2K steps (read intermediate points off
the curve rather than training separate models).

Note `L_CE` also feeds the decide step (the softmax head is what the
branching policy samples digits from), so a `ce0` model may fail through
*worse branching* rather than worse deduction — check calls/solve and the
unsound rate to attribute the effect, and cross-reference E2's decision
policies if branching turns out to be the channel.

## Sub-study D4 — Soundness pressure knobs

Two mechanisms push the model toward never returning a wrong answer:
the **asymmetric BCE** (false eliminations penalized `w+/w− = 8×` more than
false retentions, with θ_elim = 0.1 matched to that ratio) and the **CLS
conflict head** (λ_cls = 0.1). Turn each knob:

| config | bce_pos_mult / neg_mult | ratio | conflict_loss_weight |
|---|---|---|---|
| `d4_sym` | 0.5 / 0.5 | 1× | 0.1 |
| `d4_ratio2` | 1.0 / 0.5 | 2× | 0.1 |
| `baseline` | 4.0 / 0.5 | 8× | 0.1 |
| `d4_ratio32` | 16.0 / 0.5 | 32× | 0.1 |
| `d4_nocls` | 4.0 / 0.5 | 8× | 0.0 (no CLS token; conflicts detected by empty-cell test only) |

The matching eval-only θ_elim sweeps (including the "symmetric BCE +
post-hoc tuned threshold" comparison on `d4_sym`) run in E3, which owns
all inference-time operator knobs.

**Table/Figure D4** (`plots/d4_soundness.pdf`): unsound-elimination rate,
wrong-answer count, solve rate, and calls/solve vs asymmetry ratio;
`d4_nocls` as a separate row highlighting conflict-detection P/R and
search-effort impact.

---

## Run budget

~18 training configs + baseline ≈ **55 runs**, most ≤7 B200-min
(low-`L` C1 runs are much cheaper; `d1_L32` is ~15 min; the two C2 4×
rungs are ~30 min each) ≈ **7 B200-hours** total. If budget-pressed:
drop to 2 seeds for C3/C4 and `d3_ce1`, drop `d1_L1_cm4x` (keep
`d1_L1_bigdata` — it's the strongest version of the claim), never trim
D4.

## How to run (once implemented)

```bash
# 1. Sanity gate: baseline must reproduce before anything else launches.
uv run modal run --detach experiments/sudoku/run.py \
    --steps 2000 --n-train-puzzles 1000 --n-eval-puzzles 1000 --seed 0

# 2. Enumerate the sweep (prints one `modal run --detach` command per run;
#    launch them individually, not in a shell loop).
uv run python followups/arch_ablation/configs.py list

# 3. Collect results from the Modal volume and aggregate.
uv run python followups/arch_ablation/collect.py   # → results/summary.csv

# 4. Figures.
uv run python followups/arch_ablation/plot_all.py  # → plots/*.pdf
```

## TODO(worker) — implementation checklist

Training-side plumbing (all in `experiments/sudoku/`, keep flags
default-off so the existing benchmark commands are untouched):

- [ ] `run.py`: expose `--num-layers` (plumb to
      `LoopedTransformerConfig.num_layers`; needed for `d1_untied*`).
- [ ] `train.py` + `run.py`: `--supervise {all,final}` — in `_losses`, sum
      only the supervised iterations (keep the 1/n normalization consistent
      so loss magnitudes are comparable).
- [ ] `train.py`: write in-train eval points to a structured
      `<ckpt>.train_curve.jsonl` (step, correct, wrong, calls, unsound_rate,
      cls P/R) instead of stdout-only — D2/D3 learning-curve figures depend
      on this. Consider `--eval-every 50` for the D2/D3 runs.
- [ ] Deterministic checkpoint paths: add a name-override to `run.py` /
      `train()` (the current naming appends a timestamp) so every run
      lands at exactly
      `/checkpoints/followups/e1/<config>_seed<N>.pt` — this path is the
      exchange contract E2/E3 depend on (see the conventions in
      `followups/README.md`). Refuse to overwrite an existing checkpoint
      unless `--overwrite` is passed.

Experiment-side scripts (new, in this directory):

- [ ] `configs.py` — the run matrix above as data (name → flag overrides),
      with a `list` command that prints launch commands. This file is the
      single source of truth for what ran.
- [ ] `collect.py` — pull `*.eval.json` / `*.train_curve.jsonl` for
      `e1_*` checkpoints from the Modal volume, aggregate across seeds into
      `results/summary.csv` (one row per config × seed + a mean/range roll-up).
- [ ] `plot_all.py` (or per-figure scripts) — figures D1, D2, D3, D4 as
      specified above.
- [ ] Run order: sanity gate → D1-C1 (cheapest, validates plumbing) →
      D2/D3/D4 → D1 C2–C4. No handoff needed for E3: its inputs appear at
      the fixed volume paths as C1/D2/D4 runs complete.

Analysis caveats to observe when writing up:

- CM vs DM disagreement is a *finding*, not a bug — call out the
  optimizer-step confound explicitly.
- `d1_L1_*` changes the model's *deduction* power but the outer DPLL search
  can compensate with more branching; always report calls/solve alongside
  accuracy so "same accuracy, 10× more search" is visible.
- In-train eval (max_rounds 5) and final eval (max_rounds 1000) measure
  different things (deduction-tail vs full search); don't mix them in one
  plot axis.
