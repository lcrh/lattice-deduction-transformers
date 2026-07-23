# E6 — Carrying latents across solve steps

**Question.** LDT's outer loop passes *only the lattice state* between
solve steps — the transformer's hidden state is discarded after every
step. TRM/HRM do the opposite: their outer loop carries a latent embedding
(and TRM reportedly degrades badly on Sudoku without it). The paper calls
combining the two "a natural direction for future work". Two testable
readings:

1. **Sufficiency**: the lattice is an information bottleneck by design —
   if accuracy doesn't move when we add a carried latent, the explicit
   lattice state really is a sufficient search state, which sharpens the
   paper's story.
2. **Headroom**: if it *does* help (e.g. faster training, fewer forwards
   per solve, better conflict prediction), that's the cheapest known
   upgrade path and worth a follow-up note of its own.

**Design.** Minimal architectural delta to `PowersetModel`: accept an
optional carried hidden `h_carry` and blend it at loop entry (e.g.
`h = h0 + W h_carry`, zero-init `W` so training starts at baseline
behavior); return the final `h` alongside the heads.

Variants (Sudoku-Extreme, baseline 4K-step config, 3 seeds):

| config | carried across steps | notes |
|---|---|---|
| `baseline` | nothing | current behavior |
| `e6_carry` | final-loop hidden `h` | pool stores `h` per entry; reset to zeros on backfill/conflict-reset |
| `e6_carry_detach` | same, gradient-detached | isolates "extra features" from "gradient path across steps" (we don't backprop across steps anyway — this documents that explicitly) |
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

**Cost.** 4 configs (+2 aug-regime baselines) × 3 seeds × ~15 B200-min
≈ 4.5 B200-h. The most invasive experiment in this set — schedule last.

## TODO(worker)

- [ ] `looped_transformer.py`: optional `h_carry` input + `h_out` output,
      zero-init blend so default behavior is unchanged (verify: state_dict
      compatible with old checkpoints, output byte-identical when
      `h_carry=None`).
- [ ] `train.py`: pool gains an `h` buffer ([P, S, dim] — check memory at
      batch 512, dim 128: ~21 MB fp32, fine); zero on backfill; store the
      *post-step* `h` from the no-grad `dpll_step` forward (decide which
      forward's `h` is canonical — grad or no-grad — and document it).
- [ ] `dpll.py` / `solve.py`: thread `h_carry` per row; reset policy on
      conflict-reset; aug-frame handling per the note above (add
      `StepConfig.aug_per_chain` fixed-aug mode).
- [ ] `run.py`: `--carry-latent {off,on,detach,shuffle}` flag.
- [ ] Sanity gates: `off` reproduces baseline exactly; `shuffle` control
      run before interpreting `on`.
