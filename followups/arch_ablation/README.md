# E1 — Architecture & loss ablations (Sudoku-Extreme)

**Question.** The paper's headline config bundles several design choices —
a weight-tied recurrent backbone unrolled 16×, per-iteration ("deep")
supervision, an auxiliary per-cell cross-entropy `L_CE`, and asymmetric
soundness pressure in the BCE + a dedicated conflict head. Which of these
actually carry the result, and by how much?

Three specific claims/choices this experiment puts numbers on:

- **Recursion**: is a weight-tied recurrent backbone necessary at all,
  given the lattice scaffolding already makes the *outer* process
  incremental? A single-pass transformer inside the same search loop is
  the natural null hypothesis.
- **`L_CE` speeds up learning**: currently asserted in the paper without a
  supporting result.
- **Soundness pressure**: the asymmetric BCE and CLS conflict head are the
  mechanism behind the empirical-soundness claim; ablating them should
  show wrong answers reappearing.

**Testbed.** Sudoku-Extreme, 1K-puzzle train split, the Table-1 4K-step
config as baseline (800K params, ~15 min on 1× B200). Every ablation is a
one-factor deviation from this baseline — no factorial crossing.

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
    --steps 4000 --n-train-puzzles 1000 --n-eval-puzzles 1000 --seed 0
```

Baseline hyperparameters (from `experiments/sudoku/run.py` defaults):
4 layers × 16 loops, dim 128, batch 512, `bce_pos_mult/neg_mult = 4.0/0.5`
(asymmetry ratio 8×), `softmax_loss_weight = 0.2` (`L_CE`),
`conflict_loss_weight = 0.1`, θ_elim 0.1, all-iteration supervision,
final-iteration readout at inference (`use_final=True` in `dpll_step`).

**Eval protocol for the whole experiment:** fixed 1,000-puzzle test
subsample (`--n-eval-puzzles 1000`, dataset seed 200), 3 training seeds per
config. The paper's full-split eval (`eval_only.py`) is reserved for rows
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

## Sub-study D1 — Is recursion necessary? (loop count, compute- and data-matched)

Vary the unroll count `L = n_loops` at fixed 4-layer backbone. `L = 1` is
the non-recurrent ablation: a plain 4-layer transformer doing one-shot
lattice elimination inside the same outer search loop.

Two matching regimes, because "does looping help?" has two honest readings:

- **Data-matched (DM):** `steps = 4000` for every `L`. Same number of
  optimizer steps, same pool dynamics, same examples seen — but per-forward
  compute scales with `L` (train wallclock ≈ 15 min × L/16).
- **Compute-matched (CM):** `steps = 4000 × 16/L`. Total training FLOPs
  (and B200-minutes, ~15 min/run) held constant — but low-`L` runs get more
  optimizer steps and more (augmented) data epochs. That confound is
  inherent to compute-matching; reporting *both* regimes brackets it.

Warmup and the cosine schedule are step-count-*fractions* so they rescale
automatically; `max_age` stays at 100 pool steps in both regimes.

| config | n_loops | steps | regime |
|---|---|---|---|
| `d1_L1_dm` … `d1_L32_dm` | 1, 2, 4, 8, 32 | 4000 | DM |
| `d1_L1_cm` … `d1_L32_cm` | 1, 2, 4, 8, 32 | 64000, 32000, 16000, 8000, 2000 | CM |
| `baseline` | 16 | 4000 | both (shared point) |

**Untied-depth control.** Weight-tied looping ≠ plain depth. Two extra
non-recurrent (`L = 1`) runs with more *untied* layers separate "iteration
with tied weights + input re-injection" from "just make it deeper":

| config | num_layers | n_loops | steps | params |
|---|---|---|---|---|
| `d1_untied8` | 8 | 1 | 4000 | ~1.6M |
| `d1_untied16` | 16 | 1 | 4000 | ~3.2M |

(Params grow with untied depth — that's the point of the comparison; the
tied 4×16 baseline gets effective depth 64 at 800K params.)

**Figure D1** (`plots/d1_loops.pdf`): solve rate (and a second panel:
unsound-elimination rate) vs `L`, log-x, one line per regime, untied-depth
runs as scatter points at their effective depth. 3 seeds, mean ± range.

The DM checkpoints double as E3's `L_train` axis — name them so E3 can
glob them (`e1_d1_L<L>_dm_seed<N>_...`).

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

The paper asserts "Adding `L_CE` helps the model learn faster" with no
supporting result. Fix that with learning curves:

| config | softmax_loss_weight |
|---|---|
| `d3_ce0` | 0.0 |
| `baseline` | 0.2 |
| `d3_ce1` (optional) | 1.0 |

**Figure D3** (`plots/d3_ce_curves.pdf`): in-train solve count vs step +
final accuracy/soundness at 1K/2K/4K steps (read intermediate points off
the curve rather than training three separate models).

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

~16 training configs + baseline, × 3 seeds ≈ **50 runs**, most ≤15
B200-min (DM low-`L` runs are much cheaper; `d1_L32_dm` is ~30 min)
≈ **10 B200-hours** total. If budget-pressed: drop to 2 seeds for D1-CM
and `d3_ce1`, never for D4.

## How to run (once implemented)

```bash
# 1. Sanity gate: baseline must reproduce before anything else launches.
uv run modal run --detach experiments/sudoku/run.py \
    --steps 4000 --n-train-puzzles 1000 --n-eval-puzzles 1000 --seed 0

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
default-off so the paper-repro commands are untouched):

- [ ] `run.py`: expose `--num-layers` (plumb to
      `LoopedTransformerConfig.num_layers`; needed for `d1_untied*`).
- [ ] `train.py` + `run.py`: `--supervise {all,final}` — in `_losses`, sum
      only the supervised iterations (keep the 1/n normalization consistent
      so loss magnitudes are comparable).
- [ ] `train.py`: write in-train eval points to a structured
      `<ckpt>.train_curve.jsonl` (step, correct, wrong, calls, unsound_rate,
      cls P/R) instead of stdout-only — D2/D3 learning-curve figures depend
      on this. Consider `--eval-every 50` for the D2/D3 runs.
- [ ] Checkpoint naming: prefix with config id (`e1_<config>_seed<N>`), and
      keep the D1-DM names stable — E3 globs them.

Experiment-side scripts (new, in this directory):

- [ ] `configs.py` — the run matrix above as data (name → flag overrides),
      with a `list` command that prints launch commands. This file is the
      single source of truth for what ran.
- [ ] `collect.py` — pull `*.eval.json` / `*.train_curve.jsonl` for
      `e1_*` checkpoints from the Modal volume, aggregate across seeds into
      `results/summary.csv` (one row per config × seed + a mean/range roll-up).
- [ ] `plot_all.py` (or per-figure scripts) — figures D1, D2, D3, D4 as
      specified above.
- [ ] Run order: sanity gate → D1-DM (cheapest, validates plumbing) →
      D2/D3/D4 → D1-CM (most expensive). Hand the checkpoint manifest to
      E3 when D1-DM and D2 finish.

Analysis caveats to observe when writing up:

- CM vs DM disagreement is a *finding*, not a bug — call out the
  optimizer-step confound explicitly.
- `d1_L1_*` changes the model's *deduction* power but the outer DPLL search
  can compensate with more branching; always report calls/solve alongside
  accuracy so "same accuracy, 10× more search" is visible.
- In-train eval (max_rounds 5) and final eval (max_rounds 1000) measure
  different things (deduction-tail vs full search); don't mix them in one
  plot axis.
