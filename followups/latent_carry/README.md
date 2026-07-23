# E6 — Carrying latents across solve steps

**Question.** LDT's outer loop passes *only the lattice state* between
solve steps — the transformer's hidden state is discarded after every
step. TRM/HRM do the opposite: their outer loop carries a latent embedding
(and TRM reportedly degrades badly on Sudoku without it). Combining the
two is the obvious hybrid. Two testable readings:

1. **Sufficiency**: the lattice is an information bottleneck by design —
   if accuracy doesn't move when we add a carried latent, the explicit
   lattice state really is a sufficient search state, which sharpens the
   interpretability story (everything the model "knows" is legible in the
   lattice).
2. **Headroom**: if it *does* help (e.g. faster training, fewer forwards
   per solve, better conflict prediction), that's the cheapest known
   upgrade path and worth a follow-up note of its own.

**Design.** Minimal architectural delta to `PowersetModel`: accept an
optional carried hidden `h_carry` and blend it at loop entry (e.g.
`h = h0 + W h_carry`, zero-init `W` so training starts at baseline
behavior); return the final `h` alongside the heads.

**Gradient scope — the latent is pool data, not a gradient path.** The
carried `h` is written to the pool from the no-grad `dpll_step` forward
that advances the state, so it is detached by construction; no gradient
ever crosses solve steps (unchanged from the base trainer, which never
propagates gradients across solve steps). Within a step, gradient flows
through the read path (`W·h_carry`), so the model learns to *use* the
latent — but nothing gets direct credit for *writing* a latent that helps
a later step. TRM-style truncated BPTT through the carried latent is a
deliberate non-goal here (it would require keeping graphs alive across
pool iterations); it's the escalation to consider only if this cheap
version shows headroom.

Variants (Sudoku-Extreme, the E1 baseline 2K-step config — below ceiling,
so a carried latent has visible headroom to improve into; 3 seeds):

| config | carried across steps | notes |
|---|---|---|
| `baseline` | nothing | current behavior |
| `e6_carry` | final-loop hidden `h` | pool stores `h` per entry; reset to zeros on backfill/conflict-reset |
| `e6_shuffle` (control) | another puzzle's `h` | if this matches `e6_carry`, the latent is noise, not information |

Bookkeeping that must be right for the comparison to mean anything:

- The **solver** must carry `h` per chain too (reset on chain reset —
  or, interesting sub-variant, *keep* it across resets so the chain
  "remembers" its failed attempt: a learned analog of E2's dead-end
  avoidance; run only if the basic carry helps).
- Augmentation frames change per step (fresh random symmetry each
  `dpll_step`): the carried `h` lives in the *previous* step's aug frame.
  Simplest correct option: disable per-step aug re-sampling for carry runs
  (fix one aug per chain lifetime) and run a matching baseline under the
  same aug regime. Do NOT silently mix frames.

**Metrics.** Learning curves (train_curve.jsonl), final accuracy /
soundness, forwards per solve, conflict-head P/R (the latent plausibly
helps most here — counterfactual memory of what was tried).

**Deliverable.** One table + learning-curve figure; a paragraph answering
"is the lattice a sufficient outer-loop state?" either way.

**Cost.** 3 configs (+2 aug-regime baselines) × 3 seeds × ~7 B200-min
≈ 2 B200-h. The most invasive experiment in this set — schedule last.

## TODO(worker)

- [ ] `looped_transformer.py`: optional `h_carry` input + `h_out` output,
      zero-init blend so default behavior is unchanged (verify: state_dict
      compatible with old checkpoints, output byte-identical when
      `h_carry=None`).
- [ ] `train.py`: pool gains an `h` buffer ([P, S, dim] — check memory at
      batch 512, dim 128: ~21 MB fp32, fine); zero on backfill; store the
      *post-step* `h` from the no-grad `dpll_step` forward (the canonical
      writer — it's the forward that produced the state transition, and it
      keeps the latent detached by construction).
- [ ] `dpll.py` / `solve.py`: thread `h_carry` per row; reset policy on
      conflict-reset; aug-frame handling per the note above (add
      `StepConfig.aug_per_chain` fixed-aug mode).
- [ ] `run.py`: `--carry-latent {off,on,detach,shuffle}` flag.
- [ ] Sanity gates: `off` reproduces baseline exactly; `shuffle` control
      run before interpreting `on`.
