# E2 — Search-process ablations: decision policies & backtracking

> Internal planning doc — will be tightened up before this dir goes
> public-facing.

**Question.** LDT's outer loop is a stochastic DPLL: deduce → (maybe)
branch → on conflict, throw the whole chain away and restart from the
original puzzle. Both halves of that search process are currently the
simplest possible choice:

- **Decision policy** (`dpll_step`, decide block): pick a multi-alive cell
  *uniformly at random*, then softmax-sample a digit at `temp_decide=1.5`.
- **Backtracking policy** (`solve.py`, conflict handling): *reset to the
  initial state* — no partial undo, no memory of what failed. (The trainer's
  analog: true-positive-conflict pool entries are discarded and backfilled
  with fresh puzzles.)

How much performance is left on the table, and — because a core design
principle of LDT is that **the same search process runs at training and
inference** (it shapes the distribution of lattice states the model is
trained on) — do smarter policies need to be *trained with*, or can they be
swapped in at inference on an existing checkpoint?

Scope note: this experiment ablates the *decide* and *backtrack* halves of
the search loop. The *deduce* half (inference loop count, iterated
deduction-to-fixpoint, thresholds) is
[E3 `deduction_operator/`](../deduction_operator/). One planned crossover:
if E3's deduce-to-fixpoint wins at inference, its matched-training variant
runs here with the S2 machinery.

**Testbed.** Sudoku-Extreme. Primary checkpoints, referenced by their
fixed Modal-volume paths (conventions in `followups/README.md`): the E1
baseline (`/checkpoints/followups/e1/baseline_seed<N>.pt`; 2K steps,
strong deduction → little search) **and** a 1K-step checkpoint this
experiment trains and owns
(`/checkpoints/followups/e2/base_1k_seed<N>.pt`; weak deduction → lots of
search). Policy effects should be much larger on
the weak model, and that contrast is itself a finding: search heuristics
matter most when the learned deduction is imperfect, i.e. presumably on any
harder future domain.

**Key cost metric.** Parallel chains hide per-chain efficiency (64 chains
racing means a bad policy just wastes 63 chains silently). Always report,
per policy: solve rate, `puzzle_calls` (batched cost), **and** the
sequential-cost estimate (`--estimate-sequential`, already implemented in
`solve.py`) which approximates what a K=1 solver would pay. Plus resets and
decision-depth-at-conflict distributions.

---

## Sub-study S1 — Decision policies (eval-only scan)

When the model has to guess, does it matter *where* it guesses and *how*
it picks the digit — or is uniform-random cell + softmax sampling already
close to optimal?

Cheap first pass: swap the decide policy at inference on frozen
checkpoints. Two independent axes:

**Cell selection** (which multi-alive cell to branch on):

| policy | rule | notes |
|---|---|---|
| `uniform` | uniform over multi-alive cells | baseline |
| `mrv` | fewest alive candidates (min-remaining-values) | classic DPLL heuristic; needs no model output, just the state |
| `min_entropy` | lowest entropy of softmax-head distribution over alive candidates | "most decided" cell — the greedy direction |
| `max_entropy` | highest entropy | control: should be *bad*; confirms the axis matters |

**Digit selection** (which candidate to pin at the chosen cell):

| policy | rule | notes |
|---|---|---|
| `softmax@τ` | sample from softmax over alive, τ ∈ {0.5, 1.0, 1.5, 3.0} | τ=1.5 is baseline; τ→0 exists already (`temp_decide=0` → argmax) |
| `argmax` | greedy top-logit | deterministic |
| `rank_k` | chain *k* in a slot takes the k-th best digit | deterministic-but-diverse: the fix for argmax making all K chains identical |

Note the degenerate-diversity trap: a fully deterministic policy
(`min_entropy` + `argmax`) collapses all K chains onto one trajectory, so
K-way parallelism buys nothing and only the reset randomness (from
eval-time augmentation wrapping) differentiates chains. `rank_k` (and
reporting the sequential-cost estimate) is how we make the greedy
comparison fair rather than trivially bad.

Scan: ~10 selected combos (not the full cross) × {baseline ckpt, 1K ckpt}
× 1000-puzzle eval. Eval-only, ~2–4 B200-min each.

## Sub-study S2 — Matched vs. mismatched training

Does a better search policy need to be *trained with* — or can it be
bolted on at inference to a model trained under a different one?

Design assumption under test: running the identical search process at
train and inference is what keeps the transformer in-distribution. Take
the best policy P* from S1 plus the baseline policy P0 and train/eval
all four cells:

| | eval P0 | eval P* |
|---|---|---|
| **train P0** | baseline | S1 already measured this |
| **train P*** | mismatch control | the matched run |

Training with a policy means: the no-grad `dpll_step` inside `train.py`
uses that policy, so the pool's state distribution is the one the policy
induces. 2 new training configs (train-P* is the only new one, but re-run
baseline under identical eval protocol) × 3 seeds × ~7 B200-min.

Hypothesis worth falsifying: cell policy changes *which* states the model
sees (e.g. MRV visits low-branching states) → training matched to the
inference policy should win; if it doesn't, the state distribution is less
fragile than assumed, which is worth knowing (and reporting).

## Sub-study S3 — Backtracking policies

When a chain hits a conflict, must we throw away everything it did — or
can partial, stochastic, or negation-learning backtracking reuse the
work that came before the fatal decision?

Reset-to-root discards all information in a failed chain — every sound
deduction and every good decision made before the fatal one. Alternatives,
roughly in order of increasing statefulness:

| policy | on conflict, chain resets to… | extra state |
|---|---|---|
| `root` | the original puzzle | none (baseline) |
| `last` | snapshot before the most recent decision | per-chain snapshot stack |
| `geometric(p)` | snapshot j decisions up, j ~ Geometric(p); j ≥ depth → root | snapshot stack |
| `uniform_depth` | snapshot at a uniformly random earlier decision | snapshot stack |
| `last+negate` | like `last`, but additionally *kill the pinned candidate* in the restored state (classic DPLL clause-learning, depth-1 version) | snapshot stack |

Notes on `last+negate`: it is only sound if the conflict was actually
caused by that decision (and the deductions after it were sound). Because
our deductions are learned/approximate and the CLS head has false
positives, negation can eliminate the true digit and brick the chain →
must pair with a fallback (if the restored cell goes empty, escalate to
root) and we should measure how often negation kills a GT bit
(`unsound_negation_rate`, same style as the existing unsound-deduction
diagnostic).

Stochastic depth (`geometric`, `uniform_depth`) is the interesting middle
ground: it preserves a random amount of prefix work, which both amortizes
deduction *and* implicitly randomizes the restart — potentially replacing
the diversity we currently get only from full restarts.

Snapshot cost is negligible: state is [81×9] — a full per-decision
snapshot stack for a 512-row batch × ~60 decisions is ~25 MB as uint8.

**Two phases, like S1/S2:**
1. *Eval-only:* all policies on the frozen baseline + 1K checkpoints.
2. *Matched training* for the winner(s): the trainer's discard-and-backfill
   on true-positive conflict is replaced by restore-to-snapshot (the pool
   entry survives with its rolled-back state; age keeps ticking so
   `max_age` still bounds residency). This is the part that changes the
   training state distribution — e.g. `last`-style training feeds the model
   many near-conflict states, which may sharpen the CLS head, or may skew
   the pool away from fresh puzzles. Watch `pool_sat` and the age
   distribution in the train logs.

## Sub-study S4 — Policy gain vs. model strength (figure)

How much can smarter search compensate for a weaker, cheaper-trained
model?

Take the 2–3 best (policy, backtrack) combos + baseline and evaluate each
across the training-budget axis: checkpoints at 1K / 2K / 4K steps (1K
and 2K already exist at the fixed paths above; train a `base_4k` into
`/checkpoints/followups/e2/` — it's ~15 B200-min).

**Figure S4**: x = train steps, y = sequential forwards/solve (log), one
line per search config. Expectation: lines converge as deduction gets
strong (search barely happens at 4K) and fan out at 1K — quantifying "how
much can search quality compensate for training compute".

---

## Deliverables

- **Table S1**: decision-policy scan — solve rate / batched calls /
  sequential-forwards p50, p90 / resets, per checkpoint.
- **Table S2**: matched-vs-mismatched 2×2 (the train/eval distribution
  claim, tested).
- **Table S3**: backtracking policies + `unsound_negation_rate` column.
- **Figure S4**: search-quality × training-budget interaction.
- Distribution figure (reuse `plot` conventions from `repro/`):
  per-puzzle forwards histograms for baseline vs. best policy, and
  decision-depth-at-conflict histograms per backtrack policy.

## Run budget

S1 + S3-phase-1 are eval-only: ~30 evals × ~3 B200-min ≈ 1.5 B200-h.
S2 + S3-phase-2: ~4 training configs × 3 seeds × 7 B200-min ≈ 1.5 B200-h.
S4: eval-only over existing checkpoints, ~1 B200-h.

## TODO(worker) — implementation checklist

Step-operator plumbing (`experiments/sudoku/dpll.py`):

- [ ] `StepConfig.cell_policy: str = "uniform"` — implement `mrv`,
      `min_entropy`, `max_entropy` in the decide block. Entropy from the
      softmax head restricted to alive candidates (renormalized). MRV from
      alive counts only. Break ties randomly (multinomial over the argmin
      set), otherwise deterministic policies bias toward low cell index.
- [ ] `StepConfig.digit_policy: str = "softmax"` — `argmax` (can reuse
      `temp_decide=0` path), `rank_k`. `rank_k` needs the chain's rank
      within its slot: add an optional `decide_rank: Tensor | None` arg to
      `dpll_step` ([B] long, filled by `solve()` as `row_index % K`; trainer
      passes None → rank 0). Clamp rank to alive-count−1.
- [ ] Keep all new policies inert-by-default so existing repro commands
      are byte-identical.

Solver plumbing (`experiments/sudoku/solve.py`):

- [ ] Per-chain decision snapshot stack: on each decide, push (state
      pre-pin, cell, digit). Suggested layout: `snap_state [B, D_max, S, C]`
      uint8 + `snap_cell/digit [B, D_max]` + `depth [B]`, `D_max ≈ 64` with
      root-fallback on overflow.
- [ ] `SolveConfig.backtrack: str = "root"` + `geometric_p: float` +
      `learn_negation: bool` — implement the reset in the conflict-handling
      branch (currently `new_state[idx] = original[idx]`). Root stays the
      default and must remain behaviorally identical.
- [ ] Negation fallback: after killing the pinned candidate, if that cell
      is empty → escalate to root. Count `unsound_negation` events (negated
      candidate == GT digit, GT available at eval-time diagnostics same as
      unsound-deduction accounting).
- [ ] New diagnostics in `SolveResult` + eval jsonl: decision depth at
      conflict (histogram), backtrack target depth, negations, unsound
      negations. Wire into `run.py`/`eval_only.py` json output.
- [ ] `estimate_sequential` semantics under partial backtracking: an
      "attempt" currently ends on reset-to-root. Decide + document what it
      means when a chain backjumps (suggestion: attempts end only on
      root-resets; partial backjumps extend the attempt).

Trainer plumbing (`experiments/sudoku/train.py`) — S2/S3 phase 2 only:

- [ ] Trainer uses `cfg.step.cell_policy/digit_policy` automatically (it
      calls the same `dpll_step`) — verify, don't assume, that no code path
      hard-codes uniform.
- [ ] Pool-side snapshot restore for backtrack-matched training: on
      `true_positive_conflict`, restore entry to rolled-back snapshot
      instead of discard+backfill (keep discard on `max_age` and `solved`).
      Pool needs its own snapshot stack, same layout as solver's.
- [ ] Log pool health under new policies (`pool_sat`, age percentiles,
      depth distribution) — these move when the reset policy changes and
      explain training differences.

Experiment scripts (this directory, mirroring `arch_ablation/`):

- [ ] `configs.py` — S1 scan combos, S2 2×2, S3 policy list as data;
      `list` prints individual `modal run --detach` launch commands
      (eval-only ones target `eval_only.py` with a `--checkpoint` set to
      the fixed volume path of the input). Includes the `base_1k` training
      config this experiment owns. Eval outputs land under
      `/checkpoints/followups/e2/` with the `__on__` naming convention.
- [ ] `collect.py` / `plot_all.py` — summary CSV + Table S1–S3 +
      Figure S4 + distribution figures.
- [ ] Sanity gates: (1) new code with all-default flags reproduces the
      baseline eval numbers exactly; (2) `backtrack=root` through the new
      snapshot machinery matches the old code path; (3) `rank_k` with K=1
      equals `argmax`.
