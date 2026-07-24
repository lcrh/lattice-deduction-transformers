# E6 — Carrying latents across solve steps

**Question.** LDT's outer search loop passes *only the explicit lattice
state* between solve steps — the transformer's continuous hidden
activations are discarded after every step. TRM-style recursive
reasoners do the opposite: they carry continuous latents across their
outer improvement steps, and report that this matters. Does LDT need a
carried latent too, or is the explicit lattice already a sufficient
outer-loop state?

## Background: what TRM carries, and what LDT carries

**TRM** (Tiny Recursive Models, arXiv:2510.04871) improves an answer
over many outer steps by carrying two continuous embeddings between
them: a current-answer embedding `y` (decodable to a solution by the
output head) and an opaque scratchpad latent `z` (not decodable —
working memory). Within each step, the network first updates `z`
several times while looking at the puzzle input, the current answer,
and the scratchpad; then it refines `y` from the answer and scratchpad
*without* re-seeing the puzzle input. TRM argues the split is
load-bearing: drop `z` and the model forgets *how* it got to its
current answer; drop `y` and it must waste scratchpad capacity storing
the answer. Their Sudoku ablation (Table 2 of the paper) shows carrying
both features beats carrying either alone.

**LDT** carries neither. Between solve steps the only persistent state
is the lattice — per-cell candidate sets, updated *outside* the network
by hard deduce/decide operations on the model's per-step predictions.
The lattice plays the *role* of TRM's `y` (the persistent,
answer-shaped state that the next step conditions on), which is what
makes the hybrid tempting: keep the lattice, add a carried scratchpad.

**Honest limits of the analogy — stated up front so we don't oversell.**
The lattice is `y` by role only, not by mechanism. TRM's `y` is a
continuous embedding the network writes freely and can revise; it
carries soft information (confidence, near-ties) across steps. LDT's
lattice is discrete and hard-quantized every step, updated by an
external monotone operator (bits only die; only a chain reset undoes a
commitment), and it is a partial-assignment/domain store rather than a
complete proposed solution. So E6 does not reproduce TRM's `y,z`
configuration — it tests a hybrid ("discrete, externally-updated `y` +
optional continuous `z`") that TRM never ran. In particular, TRM's
Table 2 result does **not** predict E6's outcome in either direction.

## The questions

**Q1 — Is the lattice a sufficient outer-loop state?**
If we give the model a continuous memory that survives across solve
steps — anything beyond the lattice — does accuracy, search effort, or
conflict prediction improve? If not, the lattice is demonstrably a
sufficient search state, which sharpens the interpretability claim:
everything the model "knows" between steps is legible in the lattice.
If yes, that's the cheapest known upgrade path and gets its own
follow-up.

**Q2 — Recycled activation, or dedicated scratchpad?**
If a carry *does* help, does it need to be special? The cheap version
recycles the model's ordinary internal activation — the same hidden
state that feeds the prediction heads — and blends it back in at the
next step. The TRM-style version maintains a *separate* scratchpad,
architecturally distinct from the answer path: updated under different
rules (the scratchpad update sees the puzzle input; the answer refine
does not), never read by the heads, and the only thing written to the
carry. TRM argues the separation matters; this tests whether it
matters *here*.

**Q3 — Carry, or just architecture? (control)**
The dedicated-scratchpad design changes the forward pass itself: it
adds an input the current model never sees (the original puzzle givens,
which lets the model distinguish given cells from its own commitments),
an extra answer-refine application, and a restructured update schedule.
Any gain in the scratchpad arm could come from those architectural
changes rather than from actually carrying `z` across steps. A
zero-carry control — identical architecture, scratchpad forced to zero
at every step boundary — separates the two. Without it, neither a win
nor a null in the scratchpad arm is attributable.

## The four arms

| arm | carried across solve steps | forward pass |
|---|---|---|
| `baseline` | nothing (lattice only) | current model, unchanged |
| `carry_h` | final hidden `h` (the head-feeding activation) | current model + carry blended in at loop entry |
| `carry_z` | dedicated scratchpad `z` (never feeds heads) | two-stream TRM-style forward (below) |
| `zero_carry_z` | nothing — `z` zeroed at every step boundary | identical to `carry_z` |

**`carry_h`.** The pool stores the final-loop hidden state; the next
step blends it back at loop entry through a zero-initialized
projection, so with the carry absent the arm is exactly the baseline.
One stream: the carried thing *is* the ordinary activation that also
feeds the heads.

**`carry_z`.** Within one solve step's forward, the single tied
backbone maintains two streams: an answer stream `y` embedded fresh
from the current lattice, and a scratchpad `z` initialized from the
carried latent (again through a zero-initialized blend). Scratchpad
updates see the embedded original givens + answer + scratchpad; the
answer refine sees only answer + scratchpad (TRM's x-inclusion
asymmetry — the presence or absence of the puzzle input is what tells
the tied network which job it's doing). Heads read the answer stream
only; the pool stores the last scratchpad state. Role separation is
therefore enforced by what each update sees and by where each output
goes, not by separate networks.

**`zero_carry_z`.** Byte-identical forward to `carry_z`; the carry is
replaced with zeros at every step boundary. Reads on Q3: `baseline` →
`zero_carry_z` measures the architecture change; `zero_carry_z` →
`carry_z` measures the carry itself; `carry_h` vs `carry_z` then
approximately measures separation.

(A fifth "shuffle" arm — carry another puzzle's latent — was considered
and dropped; the zero-carry control answers the attribution question
more directly.)

## What must be true for the comparison to mean anything

**Gradient scope: no BPTT across solve steps.** Gradients never cross a
solve-step boundary — the carried latent is *pool data*, read like the
lattice already is, not a gradient path. In plain terms: the model is
trained, within each step, to make good predictions given whatever
carry it was handed; it is never directly rewarded for writing a carry
that helps a *future* step. This is not a handicap we're imposing — it
is exactly TRM's regime (TRM detaches its carries between outer steps
too, and its carry becomes useful anyway), and it is unchanged from the
existing LDT trainer. Because the scratchpad updates happen in-graph
within a step, the model does learn to write within-step-useful
scratchpads; cross-step usefulness has to emerge the same way it does
in TRM. Cross-step BPTT is a deliberate non-goal.

**Supervision density must be matched.** The baseline supervises the
heads at *every* internal loop iteration. A naive scratchpad forward
has only one head readout (the single answer refine), which would train
the `carry_z` arms on a fraction of the baseline's supervision signal —
a confound big enough to manufacture a null on its own. Constraint: the
`carry_z` / `zero_carry_z` forward runs several (scratchpad-loop +
answer-refine) recursions per step, with head supervision on each
recursion's answer, and with total backbone applications per forward
matched to the baseline's loop count. Report the residual readout-count
difference honestly.

**Augmentation: fixed frame per chain lifetime.** Dataset-level
augmentation stays ON (each fresh pool entry / solve chain gets one
random symmetry that persists until it is discarded); the per-step
re-augmentation that currently resamples a fresh symmetry inside every
`dpll_step` is turned OFF for all four arms. Rationale: a carried
latent lives in the embedding frame of the step that wrote it — if the
frame is reshuffled every step, the carry is incoherent by
construction. Turning *all* augmentation off would fix that too, but at
the cost of a serious training-regime shift versus E1 (few puzzles, no
symmetry diversity). Fixed-frame-per-chain keeps the diversity without
mixing frames. Consequence: arm 1 under this policy *is* the matched
baseline — no extra baseline runs needed — but its numbers are not
comparable to E1's published per-step-aug baseline, so the sanity gate
below re-establishes the reference point.

**Carry hygiene.** The carried latent is zeroed wherever its lattice
context is invalidated: pool backfill, chain conflict-reset, and slot
refill in the solver. Blends are zero-initialized. With carry absent,
`carry_h` therefore reproduces the baseline forward exactly, while
`carry_z` reproduces `zero_carry_z` exactly (the dedicated architecture
is intentionally different from baseline). Those identities are the
sanity gates for the plumbing.

## Testbed, metrics, deliverable

Sudoku-Extreme, the E1 below-ceiling budget (2K steps — deltas visible,
~7 B200-min per run), 3 seeds per arm. Sanity gate before the sweep:
re-run the baseline arm under the fixed-frame aug policy and confirm it
still sits usefully below ceiling; the headroom claim was established
under the old aug regime and must be re-verified before the carry arms
can be interpreted.

Metrics, all already emitted by the standard eval:

- learning curves (`train_curve.jsonl` — the in-train mini-solve),
- final accuracy and soundness,
- forwards per solve (search effort),
- conflict-head precision/recall — the carry plausibly helps here most
  (counterfactual memory of what was tried), so split it by fill level.

Cheap interpretive probe worth logging in the carry arms: evaluate the
trained model once with the carry zeroed at eval time — the drop (or
absence of one) measures how much information the carry actually ended
up transporting, and distinguishes "lattice was sufficient" from "the
model never learned to use the carry".

**Deliverable.** One table (4 arms × 3 seeds) + learning-curve figure,
and a paragraph answering Q1 either way — with the Q3 control used to
attribute any `carry_z` effect, and with the analogy caveat above
carried into the write-up: a null here means *this* outer loop doesn't
benefit from a carried latent (LDT's hard, stochastic decide operator
is part of the system under test), not an information-theoretic proof
that the lattice captures everything.

**Cost.** 4 arms × 3 seeds × ~7 B200-min ≈ **1.5 B200-h** (+1 run for
the sanity gate). Still the most invasive experiment in this set —
touches model, trainer, and solver — schedule last, after E1–E3 have
stabilized the shared code.

## Results

Complete (12/12). Summary write-up: [`../RESULTS.md`](../RESULTS.md#e6--carrying-latents-across-solve-steps).
Raw table: [`results/summary.csv`](results/summary.csv). Curves:
[`plots/learning_curves.png`](plots/learning_curves.png).

| arm | mean pass | wrong (sum) | calls/solve | unsound |
|---|---:|---:|---:|---:|
| baseline | 97.9% | 0 | 12.8 | 0.92% |
| carry_h | 31.7%† | 0 | 311 | 0.07% |
| carry_z | 38.8%† | 0 | 265 | 0.09% |
| zero_carry_z | 74.9%† | 413 | 37.3 | 2.46% |

†Early-abort prefixes on collapsed seeds. Continuous carry collapses search
via timeouts; the zero-carry `y`/`z` architecture alone already hurts and
introduces wrong answers. Lattice-only outer state wins.
