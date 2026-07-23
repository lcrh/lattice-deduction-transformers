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
- Report accuracy *per test order* (not pooled) — the interesting curve is
  accuracy vs. distance from the training range.
- Also report **soundness per order**: does the model abstain (conflict →
  timeout) on far-OOD sizes, or does it emit wrong answers? An accuracy
  drop with preserved soundness is a qualitatively better failure mode and
  worth a sentence in the writeup.
- Search-cost per order (calls/solve): OOD difficulty may show up as
  longer searches before it shows up as failures.

**Stretch — constraint-family transfer:** hold the lattice/grid fixed but
change the constraint pattern (e.g. train on standard snowflake groups,
test with an added ring constraint, or vice versa). Requires new CVC5
generation; scope out only if order transfer produces a clean result first.

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
