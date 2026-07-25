# Varied-topology × constraint-group OOD results

**Setup.** Independently generated `snowflake_{train,test}_varied.parquet`
(orders 4–10, ~38k unique puzzles). Every run uses translation augmentation.
The `cg_aug` arm enables `--constraint-group-embed` (random injective group
IDs resampled every sample from the full vocabulary). Matrix: 4 splits × 3
seeds × {off, on} = 24 runs. Checkpoints under `/checkpoints/followups/e4/`.

## Does 82% mean pure OOD?

**Yes.** For `e4_varied_leq6_cg_off`:

- **Train:** orders `{4,5,6}` only (500 puzzles)
- **Eval:** orders `{7,8,9,10}` only (200 puzzles / seed)

There are **no** in-distribution eval puzzles in that score. Pooled accuracy
across 3 seeds is **82.7%** (496/600). Per-order:

| test order | correct | n | accuracy |
|---|---:|---:|---:|
| 7 | 147 | 150 | 98.0% |
| 8 | 122 | 138 | 88.4% |
| 9 | 136 | 159 | 85.5% |
| 10 | 91 | 153 | 59.5% |

So one order above the train boundary is nearly solved; performance falls
with distance rather than collapsing immediately.

## Pooled summary (3 seeds)

| config | train | eval | accuracy | wrong | timeout |
|---|---|---|---:|---:|---:|
| `all_cg_off` | 4–8 | 4–8 | **99.3%** | 0.7% | 0% |
| `all_cg_aug` | 4–8 | 4–8 | **99.2%** | 0.8% | 0% |
| `shift95_cg_off` | 4–8 (95% ≤5) | 4–8 | 93.8% | 6.2% | 0% |
| `shift95_cg_aug` | 4–8 (95% ≤5) | 4–8 | **96.3%** | 3.3% | 0.3% |
| `leq6_cg_off` | 4–6 | 7–10 | **82.7%** | 12.5% | 4.8% |
| `leq6_cg_aug` | 4–6 | 7–10 | 72.2% | 11.2% | 16.7% |
| `leq5_cg_off` | 4–5 | 6–10 | ~48%† | — | — |
| `leq5_cg_aug` | 4–5 | 6–10 | ~41%† | — | — |

† `leq5` evals hit `--eval-max-timeouts 50` and aborted early (~100–150 of
200 puzzles). Treat those pooled numbers as truncated / optimistic about
coverage of the hardest tail.

## Takeaways

1. **Varied topologies keep ID performance high** (~99%), so the new data
   path is a valid in-distribution baseline.
2. **Strict order transfer without any target-order support is possible** at
   the ≤6→7–10 gap (~83% pooled; ~98% one step up).
3. **Constraint-group embeddings do not help hard OOD** under this protocol;
   they slightly help soft shift (shift95) but hurt leq6/leq5 via more
   timeouts.
4. **≤5→6–10 remains too hard** under the current search budget; failures
   are dominated by timeouts, not confident wrong answers.

## Artifacts

- `results/summary.csv`, `results/per_order.csv` (from `collect.py`)
- Modal checkpoints: `/checkpoints/followups/e4/e4_varied_*_seed{0,1,2}.*`
