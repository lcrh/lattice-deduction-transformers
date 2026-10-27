# Lattice Deduction Transformers

Reproduction code for **Lattice Deduction Transformers** —
Liam Davis, Leopold Haller, Alberto Alfarano, Mark Santolucito.
[arXiv:2605.08605](https://arxiv.org/abs/2605.08605).

This repo contains the training, inference and data-generation code for every
experiment whose numbers appear in the paper, and the scripts that rebuild each
table and figure from the collected results. The three benchmarks are
self-contained subdirectories under `experiments/` (`sudoku/`, `snowflake/`,
`maze/`); the shared package `src/ldt/` holds the looped-transformer model,
the dataset loaders, the shared evaluator and the Modal utilities.

All training and inference runs are launched on [Modal](https://modal.com/)
B200 GPUs. There is no local-GPU codepath.

### Paper results at a glance

Every table and figure in the main text comes from one or more manifests in
`paper/manifests/` (the runs) and one script (the table or figure):

| Paper | Manifests | Script (`uv run python -m ...`) |
|---|---|---|
| Figure 1(b): Sudoku-Extreme compute frontier | `table1_sudoku`, `portfolio`, `baselines` | `paper.figures.fig1_frontier` |
| Table 1: Sudoku-Extreme | `table1_sudoku`, `portfolio`, `baselines` | `paper.tables.table1_sudoku` |
| Figure 4: Sudoku train/test compute | `sudoku_compute`, `portfolio` | `paper.figures.sudoku_compute` |
| Figure 5: Snowflake puzzles | — (drawn from the generator's geometry) | `paper.figures.snowflake_puzzles` |
| Table 2: Snowflake, hard puzzles | `snowflake_extreme`, `snowflake_extreme_ood` | `paper.tables.table2_snowflake` |
| Table 3: Maze-Hard | `table3_maze`, `baselines` | `paper.tables.table3_maze` |
| Figure 6(b): Maze K-sweep | `maze_ksweep` | `paper.figures.maze_ksweep` |

Portfolio statistics in the text (Section 4.1, Appendix B) come from
`paper.portfolio` (`simulate` on the held-out set, `test` on the test split)
and the portfolio-size figure from `paper.figures.portfolio_size`. The
appendix tables come from these manifests and scripts:

| Appendix | Manifests | Script (`uv run python -m ...`) |
|---|---|---|
| C: ablations | `ablation_recurrence`, `ablation_supervision`, `ablation_lambda_ce`, `ablation_soundness` | `paper.tables.appendix <manifest>` |
| C: training-input mix | `ablation_training_inputs`, `portfolio` | `paper.tables.training_inputs` |
| D: search policies | `search` | `paper.tables.appendix search` |
| E: inference-time deduction | `deduction` | `paper.tables.appendix deduction` |
| F: latent carry | `latent_carry` | `paper.tables.appendix latent_carry` |
| G: Maze-Hard failures | `table3_maze` | `paper.figures.maze_failures` |
| H: Snowflake order transfer and training-set size | `snowflake_extreme_ood`, `snowflake_extreme` | `paper.tables.snowflake_appendix` |
| I: fine-tuned LLM | `llm_finetune` | `paper.tables.llm_finetune` |
| N: compute accounting | `table1_sudoku`, `table3_maze`, ... | `paper.compute <manifests>` |

The settings the paper describes as tuned on held-out training puzzles
(θ_CLS^eval = 0.6, λ_ce = 0.2) come from `tuning_cls_threshold` and
`tuning_lambda_ce`, evaluated on `sudoku_val`, never on the test split. The
baselines are described in [`baselines/README.md`](baselines/README.md).
Every script writes to `paper/out/`; `--copy-to DIR` also copies its output
into the paper's `figs/` directory.

### What is published, and how to regenerate it

The repo contains the code, the manifests and the collected results
(`paper/results/*.csv`, one row per run, plus the baselines' result JSONs).
Per-puzzle result rows and model checkpoints are not published. Every run
can be regenerated:

1. The manifests in `paper/manifests/` define every run behind the paper.
2. `uv run python -m paper.launch launch <manifest>` trains and evaluates
   whatever is missing on Modal.
3. `uv run python -m paper.launch collect <manifest> --rows` writes
   `paper/results/<manifest>.csv` and the per-puzzle rows
   (`paper/results/rows/<manifest>/`).
4. The table and figure scripts above rebuild every table and figure from
   those files.

The tables and figures that need per-puzzle rows (paired differences in the
appendix tables, per-order Snowflake accuracy, Figure 4, the Maze-Hard
exact-match column and failure figure, the portfolio analysis) run only
after step 3.

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
(`src/ldt/eval/`). The checkpoint path is derived from the
training settings (`/checkpoints/<benchmark>/<train hash>-seed<seed>.pt` on
the `ldt-checkpoints` volume), so relaunching a finished
config, or any config that trains the same way, skips training and only
(re)evaluates; the resolved config is stored in the checkpoint.

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
- `ablation`: the same rule, capped at 4,096 puzzles (8 blocks), so ablation
  arms are compared on a common prefix; `ablation_10k` extends a finished
  entry to 10,240 from the blocks already on disk.

Blocks do not depend on the stop rule, so a `full` eval after a
`sequential` one reuses its blocks, and rerunning an eval resumes it. To
re-evaluate a checkpoint with other settings:

```bash
uv run modal run --detach experiments/evaluate.py --checkpoint <ckpt> \
    --set "eval.stop=full eval.workers=100 solve.max_rounds=10000"
```

Training never sees evaluation puzzles; Sudoku-Extreme training pools also
exclude 10,000 held-out train puzzles (`sudoku_val`), which are reserved for
tuning defaults (the `tuning_*` manifests) and for in-training monitoring.

### Paper manifests: launching, status and collection

`paper/manifests/` lists every run behind each table and figure (one YAML
file each; eval presets in `paper/manifests/_eval.yaml`). Each entry carries
a tracker experiment ID (`exp_id`), and every run an entry key
`<exp_id>/<entry>/seed<s>`.

```bash
uv run python -m paper.launch list                         # manifests
uv run python -m paper.launch status table1_sudoku         # per run: done / running / ...
uv run python -m paper.launch launch table1_sudoku --dry-run --offline  # plan only, no network
uv run python -m paper.launch launch table1_sudoku         # submits only what is missing
uv run python -m paper.launch collect table1_sudoku --rows # -> paper/results/table1_sudoku.csv
uv run python -m paper.tables.table1_sudoku --copy-to path/to/paper/figs
uv run python -m paper.launch ledger --exp M1              # launch history
```

Status is read off the checkpoint volume (a run is done when its summary
exists), with one recursive listing per command; `--offline` uses the last
saved listing and the locally cached ledger files. `launch` refuses a dirty
working tree. Each run is submitted as `modal run --detach -n
ldt-<exp_id>-<entry>-s<seed>-<trainhash8>[-<evalhash8>]`; the entry points record
the code's git SHA in the checkpoint and eval spec (outside every hash),
print a greppable `[ldt-train]` / `[ldt-eval]` header, and append a record
with the Modal app and call IDs to `ledger/` on the volume. Tables and
figures are built from the collected CSVs by `paper/tables/*.py` and
`paper/figures/*.py`; `paper/portfolio.py` analyzes the model-portfolio
runs and `paper/compute.py` the compute accounting.

### Sudoku-Extreme (Table 1, Figure 4)

`configs/sudoku/paper.yaml` is the 4K-step recipe (1,000 training puzzles,
dataset-level augmentation, per-step augmentation at evaluation only). The
manifest `table1_sudoku` holds every LDT row of Table 1 and the results the
text cites as overrides of it: `steps{1k,2k,4k}_train_noaug` (the main rows),
`steps8k_train_noaug`, `noaug4k_evalaug` and `noaug4k` (no training
augmentation, with and without per-step augmentation at evaluation) and
`fullsplit_noaug_4k` (the whole train split). `portfolio` trains 15 seeds of
the 4K recipe (seeds 0-2 share Table 1's checkpoints), reruns their
1,000-round timeouts at 10,000 rounds, and evaluates them on 100,000
held-out training puzzles to choose a portfolio rule.

```bash
uv run python -m paper.launch launch table1_sudoku
uv run python -m paper.launch launch portfolio
uv run python -m paper.launch launch baselines        # HRM, TRM, Sotaku v2 released checkpoints
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

### Snowflake Sudoku (Table 2)

The Snowflake puzzles are generated with CVC5 via
[snowflakesudoku](https://github.com/lcrh/snowflakesudoku) (an MIT-licensed
fork of [santolucito/snowflakesudoku](https://github.com/santolucito/snowflakesudoku);
pinned commit, cloned at Modal-image build time). Every base puzzle gets its
own randomly grown connected layout, orders 4-10. Table 2 uses the hard
puzzles: minimal puzzles that naked and hidden singles cannot
solve. It trains and evaluates on orders 4-8 (`snowflake_extreme`), and the
order-transfer entries train on 4-6 or 6-8 and evaluate on the larger orders
(`snowflake_extreme_ood`).

```bash
# One-time: generate the hard puzzles (CVC5, CPU workers). Prints the split
# sizes and digests that src/ldt/eval/indices.py pins.
uv run modal run --detach experiments/snowflake/gen_data.py

uv run python -m paper.launch launch snowflake_extreme
uv run python -m paper.launch launch snowflake_extreme_ood
```

Eval rows record each puzzle's order, so per-order accuracy comes from the
rows.

### Maze-Hard (Table 3, K=1 and K=512)

`configs/maze/paper.yaml` is the K=1 recipe: 30×30, dim 192 with 2D RoPE,
20K steps; `paper_k512.yaml` supervises against 512 sampled shortest paths.
Evaluation uses θ_elim = 0.1 and the CLS threshold 0.53, calibrated on
held-out mazes, with eval-time dropout off. A maze counts as solved when the predicted path is any
shortest path; each result row also records whether it matches the reference
path. Training is unstable, so runs keep a snapshot every 1,000 steps and
Table 3 reports each seed's best checkpoint, over the step snapshots
(`*_snap_step<N>` entries) and the final checkpoint (`k1`, `k512`).
HRM and TRM are our B200 retrains, each with and without augmentation; Table 3
reports each one's better run (see [`baselines/README.md`](baselines/README.md)).

```bash
# K=512: pregenerate the K-path cache on a CPU container first
uv run modal run --detach experiments/maze/pregen.py --dataset maze_hard --k-solutions 512
uv run python -m paper.launch launch table3_maze
```

Maze runs write a resume snapshot every 1,000 steps; relaunching the same
config resumes from it.

### Maze K-sweep (Figure 6(b), 15×15 synthetic)

Each K in {1, 8, 32, 64, 128, 256, 512} is trained with 9 seeds and evaluated
on 1,000 fresh 15×15 mazes.

```bash
uv run python -m paper.launch launch maze_ksweep
```

---

## Directory layout

```
src/ldt/
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
baselines/            HRM/TRM evals and Maze-Hard retrains, Sotaku v2 eval, EqR
                      throughput, fine-tuned Qwen, frontier LLMs
                      (see baselines/README.md)
paper/                Manifests (runs per table/figure), launcher, tables,
                      figures, portfolio and compute analysis
tools/                inspect_wrong.py: print puzzles a run got wrong
tests/                CPU tests, including bit-exact equivalence fixtures
```

All hyperparameters are in `configs/` and in the paper's appendix.
