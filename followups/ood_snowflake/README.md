# E4 — Out-of-distribution generalization: Snowflake order transfer

**Question.** Does LDT learn *the constraint semantics* or just the puzzle
distribution it was trained on? Test by training on small Snowflake Sudoku
orders and evaluating on strictly larger, never-seen orders.

**Why this is well-posed here.** The Snowflake setup already embeds every
puzzle into a fixed 15×10 covering grid (supports up to order 19) with a
per-cell in-puzzle mask, so a single model can be evaluated on sizes it
never saw — no architecture change needed. The existing dataset spans
orders 4–8; larger orders just need more CVC5 generation.

**Design.**

| config | train orders | test orders |
|---|---|---|
| `e4_leq5` | {4, 5} | {6, 7, 8} (+ 9, 10 stretch) |
| `e4_leq6` | {4, 5, 6} | {7, 8} (+ 9, 10) |
| `e4_all` (control) | {4–8} | {4–8} held-out split (the paper setting, re-run) |

- Match `--n-train-puzzles 500` and hyperparameters to the paper's
  Snowflake config; only the order filter varies. 3 seeds.
- **Positional confound — must be fixed before any transfer run.**
  Snowflake placement is deterministic and hub-centered
  (`cell_to_grid_idx` in `experiments/snowflake/data.py`), and snowflake
  training has *no positional augmentation* (digit-perm only; dihedral is
  off for hex). Snowflakes grow outward with order, so training on small
  orders leaves the outer covering-grid positions seen only as "absent" —
  their learned 2D positional embeddings would be effectively untrained
  when order 7–8 activates them, and a transfer failure would measure
  that, not constraint generalization. Mitigations, in order of
  preference: (a) random `(q, r)` translation aug — shift input, solution,
  and in-puzzle mask together; sound because constraint groups are a
  function of the visible geometry, which translation preserves; (b) hex
  D6 rotation/reflection aug (needs a custom covering-grid cell
  permutation); (c) a 2D-RoPE variant (`use_rope` exists in the model
  config, used for 30×30 maze) as the relative-position control. Run the
  in-distribution control `e4_all` under the same aug regime as the
  transfer runs. Vocabulary needs no such care: `VOCAB = 6` at every
  order (all-different groups cap at 6 cells), so channels are identical
  across orders — only active positions and group counts change.
- Report accuracy *per test order* (not pooled) — the interesting curve is
  accuracy vs. distance from the training range.
- Also report **soundness per order**: does the model abstain (conflict →
  timeout) on far-OOD sizes, or does it emit wrong answers? An accuracy
  drop with preserved soundness is a qualitatively better failure mode and
  worth a sentence in the writeup.
- Search-cost per order (calls/solve): OOD difficulty may show up as
  longer searches before it shows up as failures.

**Why order transfer is the right OOD test (and constraint-family swap is
not).** LDT never sees a description of the rules: constraints are learned
from (puzzle, solution) pairs and live in the weights; the input carries
only the lattice state and the topology mask. Order transfer is well-posed
because the constraint *family* is fixed and the concrete constraint
groups are a deterministic function of the visible topology — the model
has everything it needs to apply the same pattern at unseen sizes. Naively
swapping the constraint family at test time is ill-posed: the change is
invisible in the input, so any method would fail, and LDT fails in the
worst direction (added constraints → confidently returns old-rules
solutions as "solved"; removed constraints → eliminations become unsound).
Do not run that version. The well-posed variant — encode constraint-group
structure *in the input*, train over a distribution of families, test on
held-out families (a learned generic AllDifferent propagator) — is a new
architecture axis and belongs in its own future experiment, not here.

**Deliverable.** One table (per-order accuracy/soundness for each training
range) + one figure (accuracy & soundness vs. test order, vertical line at
the training boundary).

**Cost.** ~9 training runs × ~5 B200-min + CVC5 generation for orders 9–10
(CPU, parallel — the existing `gen_data.py` fans out to ~100 workers).

## TODO(worker)

- [ ] `experiments/snowflake/data.py`: add order filtering
      (`orders: list[int] | None` in the dataset config) for both train and
      eval loaders; verify the covering-grid embedding is genuinely
      order-independent (no order-derived normalization anywhere).
- [ ] Occupancy check (cheap, do first): compute per-order covering-grid
      occupancy from the dataset and quantify how many positions are
      active at orders 7–8 but never active at ≤6 — this measures the
      size of the positional confound before designing around it.
- [ ] Translation augmentation: random `(q, r)` shift applied jointly to
      x / y / in_puzzle_mask, bounds-checked against the 15×10 grid;
      train-time (dataset-level or per-step, mirroring how sudoku handles
      dihedral). Hex D6 rotation aug as a follow-on if translation alone
      leaves outer positions under-covered.
- [ ] Optional RoPE control: `--use-rope` plumbed through
      `experiments/snowflake/run.py` (config support already exists in
      `LoopedTransformerConfig`); one config to compare learned-absolute
      + translation-aug vs. relative encodings.
- [ ] `experiments/snowflake/run.py`: expose `--train-orders` /
      `--eval-orders` (comma-separated), and make eval report per-order
      breakdowns in `eval.json` (`per_order: {order: {correct, wrong,
      timeout, calls}}`).
- [ ] `gen_data.py`: parameterize max order; generate orders 9–10
      (~1,000 base solutions each, same greedy-minimization uniqueness
      pipeline). Record generation cost — feeds E7.
- [ ] Sanity gate: `e4_all` must reproduce the paper's 100/100 before the
      transfer runs are interpreted.
- [ ] `collect.py` / plot script for the accuracy-vs-order figure.
