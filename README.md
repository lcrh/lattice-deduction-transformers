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

The shared package `src/lattice_diffusion/` holds the looped-transformer model,
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

Every run is described by an experiment config: a YAML file under
`configs/` plus optional `key.path=value` overrides, merged over the typed
defaults in `experiments/<benchmark>/config.py` (unknown keys are errors).
Each benchmark has one Modal entry point:

```bash
uv run modal run --detach experiments/<benchmark>/run.py \
    --config configs/<benchmark>/<name>.yaml --set "train.seed=1 eval.stop=full"
```

`--detach` lets the run survive your laptop sleeping. The run trains a
checkpoint, then evaluates it with the shared evaluator
(`src/lattice_diffusion/eval/`). The checkpoint path is derived from the
training config (`/checkpoints/<benchmark>/<name>-<hash>-seed<seed>.pt` on
the `lattice-diffusion-checkpoints` volume), so relaunching a finished
config skips training and only (re)evaluates; the resolved config is stored
in the checkpoint.

**Evaluation.** The evaluator walks a fixed index list in blocks across
B200 workers (`eval.workers`, at most `eval.max_in_flight` blocks ahead of
the finished prefix). Results go to
`/checkpoints/eval/<checkpoint>/<tag>-<hash>/`: one JSONL file per block
(one row per puzzle) and a `summary-*.json` per stop rule. The stop rule
(`eval.stop`) is one of

- `sequential`: walk a random ordering of the full test split and stop once
  the error rate is pinned down (100 failures, or a 95% confidence-sequence
  half-width of 0.1 points, after at least 1,000 puzzles);
- `full`: every puzzle in the list;
- `ablation`: exactly the first 1,000 puzzles, so ablation arms are scored
  on the same puzzles.

Blocks do not depend on the stop rule, so a `full` eval after a
`sequential` one reuses its blocks, and rerunning an eval resumes it. To
re-evaluate a checkpoint with other settings:

```bash
uv run modal run --detach experiments/evaluate.py --checkpoint <ckpt> \
    --set "eval.stop=full eval.workers=100 solve.max_rounds=10000"
```

Training never sees evaluation puzzles; Sudoku-Extreme training pools also
exclude 10,000 held-out train puzzles (`sudoku_val`), which are reserved for
choosing defaults and for in-training monitoring.

### Paper manifests: one command per table

`paper/manifests/` lists every run behind each table and figure (one YAML
file each; eval presets in `paper/manifests/_eval.yaml`). Each entry carries
a tracker experiment ID (`exp_id`), and every run an entry key
`<exp_id>/<entry>/seed<s>`.

```bash
uv run python -m paper.launch list                         # manifests
uv run python -m paper.launch status table1_sudoku         # per run: done / running / ...
uv run python -m paper.launch launch table1_sudoku --dry-run
uv run python -m paper.launch launch table1_sudoku         # submits only what is missing
uv run python -m paper.launch collect table1_sudoku --rows # -> paper/results/table1_sudoku.csv
uv run python -m paper.tables.table1_sudoku --copy-to ../latticeDiffusion/figs
uv run python -m paper.launch ledger --exp M1              # launch history
```

Status is read off the checkpoint volume (a run is done when its summary
exists); `--offline` uses the last saved listing. `launch` refuses a dirty
working tree. Each run is submitted as `modal run --detach -n
ldt-<exp_id>-<entry>-s<seed>-<trainhash8>[-<evalhash8>]`; the entry points record
the code's git SHA in the checkpoint and eval spec (outside every hash),
print a greppable `[ldt-train]` / `[ldt-eval]` header, and append a record
with the Modal app and call IDs to `ledger/` on the volume. Tables and
figures are built from the collected CSVs by `paper/tables/*.py` and
`paper/figures/*.py`; `paper/portfolio.py` analyzes the model-portfolio
runs and `paper/compute.py` the compute accounting.

### Sudoku-Extreme (Table 1, Figure 2)

`configs/sudoku/paper.yaml` is the 4K-step recipe (1,000 training puzzles).
The other rows and the step sweep of Figure 2 are overrides of it.

```bash
# 4K steps, every test puzzle
uv run modal run --detach experiments/sudoku/run.py --config configs/sudoku/paper.yaml \
    --set "eval.stop=full eval.workers=100"

# 1K and 2K steps
uv run modal run --detach experiments/sudoku/run.py --config configs/sudoku/paper.yaml \
    --set "train.steps=1000"

# No augmentation
uv run modal run --detach experiments/sudoku/run.py --config configs/sudoku/paper.yaml \
    --set "train.step.augment=false solve.step.augment=false \
           data.augment_digit_perm=false data.augment_dihedral=false"
```

`configs/sudoku/ablation.yaml` is the base for the appendix ablations (2K
steps, `ablation` stop rule). Each ablation is one override of it:

| Ablation | Options |
|---|---|
| Recurrence / shape | `model.n_loops`, `model.num_layers`, `model.dim` (a static transformer is `n_loops=1`) |
| Deep supervision | `train.supervise=final` |
| Loss terms | `train.softmax_loss_weight` (λ_CE), `train.bce_pos_mult` / `bce_neg_mult`, `train.conflict_loss_weight=0` with `model.cls_token=false` |
| Latent carry | `model.carry=h\|z\|zero_z` (with `train.step.augment=false solve.step.augment=false`) |
| Deduction at inference | `solve.step.n_loops`, `solve.step.deduce_passes` (int or `fixpoint`), `solve.step.threshold`, `solve.step.cls_threshold` |
| Search | `solve.step.cell_policy`, `solve.step.digit_policy`, `solve.backtrack`, `solve.backtrack_negate`; the same `step` options under `train.step` change the training rollouts |

Eval-only ablations of an existing checkpoint go through
`experiments/evaluate.py --set "solve...."`.

> **Note.** As reported, these runs match the Sudoku-Extreme solve rate in
> Table 1 (100%). We have since realized, however, that the reported figure
> came from an evaluation that inadvertently skipped a full pass over the
> test set. On a complete evaluation the true solve rate is ~99.96%, and we
> are in the process of correcting the paper. See
> [this thread](https://x.com/biosemiote/status/2061848053741687016) for
> context.

### Snowflake Sudoku (Table 2, order-transfer appendix)

The Snowflake puzzles are generated with CVC5 via
[snowflakesudoku](https://github.com/lcrh/snowflakesudoku) (an MIT-licensed
fork of [santolucito/snowflakesudoku](https://github.com/santolucito/snowflakesudoku)
that adds varied layouts; pinned commit, cloned at Modal-image build time).
The paper uses the varied-topology dataset: every base puzzle gets its own
randomly grown connected layout, orders 4–10. Table 2 trains and evaluates on
orders 4–8; the order-transfer configs hold out larger orders.

```bash
# One-time: generate the dataset (CVC5, ~100 CPU workers, parallel). Writes
# snowflake_varied_{train,val,test}.parquet and prints index-list values to pin.
uv run modal run --detach experiments/snowflake/gen_data.py

# Table 2: train on orders 4-8, evaluate on the order 4-8 test puzzles
uv run modal run --detach experiments/snowflake/run.py --config configs/snowflake/paper.yaml

# Order transfer: train on 4-6, evaluate on 7-10 (and 6-8 -> 9-10, and 4-6
# with 12/25/50 of the 500 training puzzles drawn from orders 7-10)
uv run modal run --detach experiments/snowflake/run.py --config configs/snowflake/ood_4to6_eval7to10.yaml
uv run modal run --detach experiments/snowflake/run.py --config configs/snowflake/ood_6to8_eval9to10.yaml
uv run modal run --detach experiments/snowflake/run.py --config configs/snowflake/ood_4to6_mixin25.yaml
```

Eval rows record each puzzle's order, so per-order accuracy comes from the
rows. The submitted paper's canonical-topology dataset (one concentric layout
per order, orders 4–8) is still available with
`gen_data.py --topology canonical --n-max 8` and `--set "data.dataset=canonical
data.translate_aug=false eval.index_list=snowflake_test"`.

### Maze-Hard (Table 3, K=1 and K=512)

`configs/maze/paper.yaml` is the K=1 recipe: 30×30, dim 192 with 2D RoPE,
20K steps. The evaluation CLS threshold 0.53 was calibrated on held-out
mazes (0.6 elsewhere), and eval-time dropout is off. A puzzle counts as
solved when the predicted path is a valid shortest path; each result row
also records whether it matches the reference path.

> **Note.** The HRM and TRM baselines we compare against on Maze-Hard run
> ~7pp higher than the figures in their original papers. This is an
> evaluation-criteria mismatch: the HRM/TRM papers report *exact match*
> against the single shortest path emitted by their (deterministic) A\*
> solver, whereas a 30×30 maze admits many equally-optimal paths, and a
> *lenient* "any valid optimal path" criterion scores ~7pp higher. We are
> reconciling this and will update the paper. See
> [`baselines/hrm_trm/README.md`](baselines/hrm_trm/README.md) for the full independent HRM/TRM
> reproduction — the lenient metric, released-checkpoint evals, and a
> with/without-augmentation training study.

```bash
uv run modal run --detach experiments/maze/run.py --config configs/maze/paper.yaml

# K=512: pregenerate the K-path cache on a CPU container first
uv run modal run --detach experiments/maze/pregen.py --dataset maze_hard --k-solutions 512
uv run modal run --detach experiments/maze/run.py --config configs/maze/paper_k512.yaml
```

Maze runs write a resume snapshot every 1,000 steps; relaunching the same
config resumes from it.

### Maze K-sweep figure (Figure 3, 15×15 synthetic)

Each K is averaged over 4 seeds; evaluation is on a seed-200 pool of 1,000
fresh 15×15 mazes.

```bash
uv run python -m paper.launch launch fig3_maze_ksweep   # K in {1, 8, ..., 512} x 4 seeds
```

---

## Directory layout

```
src/lattice_diffusion/
  data/               Streaming datasets (sudoku_extreme, maze_hard, maze_synthetic)
  models/             Looped transformer + 2D transformer backbone + weighted-BCE loss
  modal/              Shared Modal image / volume / secret definitions
  training/utils/     Checkpoint I/O and cosine LR scheduler
  eval/               Shared parallel evaluator: index lists, stop rules,
                      block scheduler, statistics, Modal coordinator/workers
  config.py           YAML + override loading for the experiment configs
configs/              Experiment configs (one directory per benchmark)
experiments/
  common.py           Experiment config base class and the train-then-eval flow
  evaluate.py         Re-evaluate any checkpoint
  sudoku/             The shared `dpll_step` deduction primitive, the
                      pool-based trainer, and the solver. All other
                      experiments import dpll/solve/ema/aug from here.
  snowflake/          Snowflake data loader + trainer + CVC5 data generator.
  maze/               Maze data loader (the `maze_hard` HF split and
                      on-the-fly `synthetic` mazes, plus a K-paths sampler
                      with a pregen cache) + trainer.
  <benchmark>/config.py     typed experiment config (defaults = paper recipe)
  <benchmark>/eval_task.py  adapter for the shared evaluator
baselines/            HRM/TRM reproduction and evals, fine-tuned Qwen baseline
                      (see baselines/README.md)
paper/                Manifests (runs per table/figure), launcher, tables,
                      figures, portfolio and compute analysis
tools/                inspect_wrong.py: print puzzles a run got wrong
tests/                CPU tests, including bit-exact equivalence fixtures
```

All hyperparameters are in `configs/` and in the paper's appendix.
