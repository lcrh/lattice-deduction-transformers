# Lattice Deduction Transformers

Reproduction code for **Lattice Deduction Transformers** —
Liam Davis, Leopold Haller, Alberto Alfarano, Mark Santolucito.
[arXiv:2605.08605](https://arxiv.org/abs/2605.08605).

This repo contains the training / inference / data-generation code for every
experiment whose numbers appear in the paper. The three benchmarks are
self-contained subdirectories under `experiments/`:

| Benchmark | Subdir | Paper result |
|---|---|---|
| Sudoku-Extreme | `experiments/sudoku/` | Table 1 (LDT rows) + Figure 2 |
| Snowflake Sudoku | `experiments/snowflake/` | Table 2 |
| Maze-Hard (30×30) | `experiments/maze/` | Table 3 + Figure 3 (K-sweep) |

The shared package `lattice_diffusion/` holds the looped-transformer model,
the dataset loaders, and the Modal image / checkpoint utilities used by all
three.

All training and inference runs are launched on [Modal](https://modal.com/)
B200 GPUs. There is no local-GPU codepath.

> **Status (reconstruction).** This is a curated reconstruction of the
> codebase used for the paper's experiments — the original development
> tree carried many exploratory branches, sweeps, and dead ends; this
> repo is the minimal subset needed to reproduce the reported numbers.
> We are currently in the process of validating the results end-to-end
> against the original runs, and will update this note as that work
> completes.

---

## Setup

```bash
uv sync
modal token new                              # one-time Modal auth
modal secret create huggingface-secret HF_TOKEN=<your HF token>
```

The HF token is read by Modal containers so the runtime `huggingface_hub`
client doesn't hit unauthenticated rate limits when downloading
`sapientinc/sudoku-extreme` and `sapientinc/maze-30x30-hard-1k`. Both
datasets are public; consult their HuggingFace dataset cards for terms
of use.

---

## Reproducing the paper results

Every experiment is a Modal app with a `local_entrypoint`. Run it with
`uv run modal run --detach <path>` (the `--detach` lets the run survive
your laptop sleeping). Each run trains a checkpoint, evaluates it, and
writes a `<ckpt>.eval.json` + `<ckpt>.eval.jsonl` next to the checkpoint
on the `lattice-diffusion-checkpoints` Modal volume.

### Sudoku-Extreme (Table 1)

The four LDT rows of Table 1 are the same `run.py` at four configurations.
The `run.py` default `--n-eval-puzzles 200` is the inline quick-check; pass
`--n-eval-puzzles 4000` to evaluate on a larger test sample, or re-eval a
saved checkpoint on the full test split with the parallel `eval_only.py`
script.

> **Note.** As reported, these runs match the Sudoku-Extreme solve rate in
> Table 1 (100%). We have since realized, however, that the reported figure
> came from an evaluation that inadvertently skipped a full pass over the
> test set. On a complete evaluation the true solve rate is ~99.96%, and we
> are in the process of correcting the paper. See
> [this thread](https://x.com/biosemiote/status/2061848053741687016) for
> context.

```bash
# LDT, 4K train steps (Table 1 top row)
uv run modal run --detach experiments/sudoku/run.py \
    --steps 4000 --n-train-puzzles 1000 --n-eval-puzzles 4000

# LDT, 1K and 2K train steps (Table 1 mid-rows)
uv run modal run --detach experiments/sudoku/run.py --steps 1000 --n-train-puzzles 1000 --n-eval-puzzles 4000
uv run modal run --detach experiments/sudoku/run.py --steps 2000 --n-train-puzzles 1000 --n-eval-puzzles 4000

# No-augmentation variant (last row)
uv run modal run --detach experiments/sudoku/run.py \
    --steps 4000 --n-train-puzzles 1000 --n-eval-puzzles 4000 \
    --no-augment --no-data-augment-digit-perm --no-data-augment-dihedral
```

For the train-vs-test-compute figure (Figure 2a), repeat the above at
multiple step counts and across 3 seeds.

### Snowflake Sudoku (Table 2)

The Snowflake puzzles are generated with CVC5 via the
[snowflakesudoku](https://github.com/santolucito/snowflakesudoku) helper
package (MIT-licensed, cloned at Modal-image build time).

```bash
# One-time: generate the ~30K-puzzle dataset (CVC5, ~100 CPU workers, parallel)
uv run modal run --detach experiments/snowflake/gen_data.py

# Train + eval (headline: 100 / 100 on 1,000-puzzle test split)
uv run modal run --detach experiments/snowflake/run.py \
    --steps 1000 --n-train-puzzles 500 --n-eval-puzzles 1000
```

`gen_data.py` writes `/data/snowflake_train.parquet` and
`/data/snowflake_test.parquet` to the `lattice-diffusion-data` volume.

### Maze-Hard (Table 3, K=1 and K=512)

`eval_cls_threshold=0.53` is the per-paper Maze-Hard CLS threshold,
calibrated on a held-out set (vs.\ the `0.6` used elsewhere).
`eval_dropout_p=0.0` disables eval-time MHA dropout for the 30×30 setting.

The 30×30 training command leaves the in-train final eval at the
default 200-puzzle quick-check (eval at the training batch size of 192
runs comfortably here; pushing it to the full 1,000 puzzles inline at
this grid + model size has been observed to OOM). The headline
1,000-puzzle number comes from the parallel `eval_only.py` call below.

```bash
# K=1 (head-to-head with TRM)
uv run modal run --detach experiments/maze/run.py \
    --dataset maze_hard --steps 20000 --batch-size 192 \
    --model-dim 192 --use-rope --pool-size-mult 2.0 \
    --k-solutions 1 --eval-cls-threshold 0.53 --eval-dropout-p 0.0

# K=512
uv run modal run --detach experiments/maze/run.py \
    --dataset maze_hard --steps 20000 --batch-size 192 \
    --model-dim 192 --use-rope --pool-size-mult 2.0 \
    --k-solutions 512 --eval-cls-threshold 0.53 --eval-dropout-p 0.0
```

For the K=512 setting the K-paths precompute is expensive; pregen the
canonical pool first on a CPU container (grid-size and worker-count are
fixed for `maze_hard` since the HF split is exactly 30×30 / single-pool;
they only matter for `--dataset synthetic`):

```bash
uv run modal run --detach experiments/maze/pregen.py \
    --dataset maze_hard --k-solutions 512
```

Then run the full 1,000-puzzle eval via the parallel `eval_only.py`
(fans across N B200 workers at a smaller per-worker batch so the
30×30 dim=192 model stays well under the GPU memory ceiling;
checkpoints land at `/checkpoints/maze/maze_hard_30x30_seed<N>_<...>.pt`):

```bash
uv run modal run --detach experiments/maze/eval_only.py \
    --checkpoint /checkpoints/maze/maze_hard_30x30_seed0_<...>.pt \
    --workers 20 --batch-size 64 --dataset maze_hard --n-eval 1000 \
    --cls-threshold 0.53 --dropout-p 0.0
```

### Maze K-sweep figure (Figure 3, 15×15 synthetic)

The K-sweep uses the base maze setup (no RoPE, dim=128) but overrides the
`maze/run.py` default deduce `--threshold 0.5` back to the dpll-default
`0.1`. Each K is averaged over 4 seeds.

```bash
# Repeat for K in {1, 8, 32, 64, 128, 256, 512} and seeds in {0, 2, 3, 4}
uv run modal run --detach experiments/maze/run.py \
    --dataset synthetic --grid-size 15 \
    --steps 4000 --batch-size 256 \
    --threshold 0.1 --n-puzzles 10000 \
    --k-solutions <K> --seed <S>
```

---

## Directory layout

```
src/lattice_diffusion/
  data/               Streaming datasets (sudoku_extreme, maze_hard, maze_synthetic)
  models/             Looped transformer + 2D transformer backbone + weighted-BCE loss
  modal/              Shared Modal image / volume / secret definitions
  training/utils/     Checkpoint I/O and cosine LR scheduler
experiments/
  sudoku/       Owns the shared `dpll_step` deduction primitive, the
                      pool-based trainer, and the parallel solver. All other
                      experiments import dpll/solve/ema/aug from here.
  snowflake/    Snowflake-specific data loader + trainer wrapper + CVC5
                      data generator.
  maze/         Maze data loader (handles both the `maze_hard` HF
                      split and on-the-fly `synthetic` mazes, plus a
                      K-paths sampler with a pregen cache) + trainer
                      wrapper + parallel eval.
```

All hyperparameters live in the `run.py` defaults and in
Appendix C of the paper. The `--help` of each `modal run …` shows every
flag.
