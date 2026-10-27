# Baselines

Where every baseline number in the paper comes from, and the scripts that
produce it. The `baselines` and `llm_finetune` manifests list the exact
`modal run` command behind each reported result:

```bash
uv run python -m paper.launch launch baselines      # HRM, TRM and Sotaku v2 evals
uv run python -m paper.launch launch llm_finetune   # fine-tuned Qwen evals
```

| Dir | Baseline | Paper |
|---|---|---|
| [`hrm_trm/`](hrm_trm/README.md) | HRM, TRM | Table 1 (Sudoku-Extreme), Table 3 (Maze-Hard) |
| [`sotaku/`](sotaku/modal_eval.py) | Sotaku v2 | Table 1 |
| [`eqr/`](eqr/modal_throughput.py) | Equilibrium Reasoners | Table 1 (accuracy as reported; B200 throughput measured here) |
| [`qwen_sft/`](qwen_sft/run.py) | Qwen3.5-0.8B, full fine-tune | Appendix I |
| [`frontier_llm/`](frontier_llm/run.py) | Frontier LLMs via OpenRouter | Appendix J |

## Sudoku-Extreme (Table 1)

HRM, TRM and Sotaku v2 are their released checkpoints, evaluated on the
first 10,000 puzzles of LDT's `sudoku_test_perm` ordering
(`ldt.eval.indices`), so every method sees the same puzzles in the same
order. A puzzle counts as solved when the output grid equals the solution.

| Baseline | Checkpoint | Result | Accuracy |
|---|---|---|---|
| HRM | `sapientinc/HRM-checkpoint-sudoku-extreme` (authors' release) | `hrm_trm/results/hrm_sudoku.json` | 55.4% |
| TRM-MLP | `alphaXiv/trm-model-sudoku/step_32550_sudoku_epoch50k` (alphaXiv reproduction; the TRM authors released no weights) | `hrm_trm/results/trm_mlp_sudoku.json` | 79.5% |
| TRM-attention | `alphaXiv/trm-model-sudoku/step_32550_sudoku_50k_epoch_attn_type` | `hrm_trm/results/trm_att_sudoku.json` | 70.4% |
| Sotaku v2 | release v2.0.0, `model_late_state_ce.pt` (commit `d0ab125`; FP32, batch 256, 1,024 iterations) | `sotaku/results/sotaku_v2_sudoku.json` | 98.96% |

The paper's TRM row is TRM-MLP. EqR's accuracy (99.8%) is as reported by its
authors; `eqr/modal_throughput.py` measures the B200 training and inference
throughput of its released config and checkpoint for the compute estimates.
HRM, TRM and Sotaku v2 training times are likewise estimated from measured
B200 throughput (`hrm_trm/trm_eval/modal_train.py::run_throughput`,
`hrm_trm/hrm_eval/modal_train.py::run_throughput`,
`sotaku/modal_eval.py::run_throughput`).

## Maze-Hard (Table 3)

HRM and TRM are our retrains on one B200 each, with the published Maze-Hard
recipes (global batch 768), once with 8x dihedral augmentation and once
without (`hrm_trm/train_with_aug.sh`, `hrm_trm/train_no_aug.sh`). Every saved
checkpoint of each run is evaluated on all 1,000 test mazes with two
metrics: *lenient* (any shortest path from start to goal) and *exact* (the
dataset's reference path). Table 3 reports the lenient metric, at each
method's best run and checkpoint by lenient test accuracy:

| Run | Sweep | Best lenient | Step | Exact at that step |
|---|---|---|---|---|
| **HRM, with augmentation** (reported) | `hrm_trm/results/hrm_aug_sweep.json` | **84.6%** | 122,388 | 16.7% |
| HRM, without augmentation | `hrm_trm/results/hrm_noaug_sweep.json` | 82.2% | 39,060 | 72.4% |
| **TRM, without augmentation** (reported) | `hrm_trm/results/trm_noaug_sweep_lcycles4.json` | **91.9%** | 19,530 | 84.5% |
| TRM, with augmentation | `hrm_trm/results/trm_aug_sweep_lcycles4.json` | 21.3% | 6,510 | 4.1% |

TRM is evaluated with the loop count it was trained with (`L_cycles=4`,
`hrm_trm/trm_eval/cfg_maze.yaml`). Checkpoints are on the Modal volumes
`hrm-maze-aug-ckpts` and `trm-maze-aug-ckpts` under
`maze-hard-repro/<run>/step_<N>`. The training times in Table 3 are the
reported step count times the measured steady-state B200 throughput (HRM about
0.61 s/step, TRM about 1.46 s/step), excluding compile and startup.

## Fine-tuned LLM (Appendix I)

`qwen_sft/run.py` fine-tunes all parameters of Qwen3.5-0.8B in BF16 for 16
epochs (seed 0, one checkpoint per epoch) on a seed-42 draw of 1,000 puzzles
from the Sudoku-Extreme train split, either as given (`natural`) or rebuilt
from each solution with exactly K blank cells (`--n-blanks K`). The
evaluated runs are on the `ldt-checkpoints` volume under
`/checkpoints/followups/llm_baseline/`:

| Run | Puzzles | Training (1x B200) |
|---|---|---|
| `qwen3_5_0_8b_natural_ep16_seed0` | natural (about 56 blanks) | 406 s |
| `qwen3_5_0_8b_blanks1_ep16_seed0` | 1 blank | 732 s |
| `qwen3_5_0_8b_blanks2_ep16_seed0` | 2 blanks | 395 s |
| `qwen3_5_0_8b_blanks4_ep16_seed0` | 4 blanks | 407 s |
| `qwen3_5_0_8b_blanks8_ep16_seed0` | 8 blanks | 767 s |
| `qwen3_5_0_8b_blanks16_ep16_seed0` | 16 blanks | 545 s |
| `qwen3_5_0_8b_blanks32_ep16_seed0` | 32 blanks | 401 s |

`run.py::evaluate` scores every second epoch's checkpoint on the first 256
puzzles of `sudoku_test_perm` (or their K-blank versions), 32 samples per
puzzle, with unbiased pass@k; the summaries are in `qwen_sft/results/`.

## Frontier LLMs (Appendix J)

`frontier_llm/run.py` sends the appendix prompts to the models in its
`MODELS` list through OpenRouter (`OPENROUTER_API_KEY`), with reasoning effort
"high", at most 100,000 output tokens and one hour per call. The puzzles, the
first 50 of each benchmark's test ordering, are in `frontier_llm/puzzles.json`
(`run.py export` rebuilds it from the datasets). Answers, token counts and
costs are in `frontier_llm/results/results.json`; `frontier_llm/analyze.py`
prints the solve rate, output tokens and cost per puzzle and classifies every
unsolved answer. The April 2026 zero-shot results in the same appendix are
reported as run at the time; their scripts are not part of this repository.

## Per-puzzle rows

The HRM, TRM, Sotaku v2 and Qwen evals write per-puzzle rows to
`eval/baselines/<model>/...` on the `ldt-checkpoints` volume, with the field
names of `ldt/eval/records.py` (`idx`, `pos`, `status`, `solution`). Every
eval saves its work in units and skips finished units when rerun: Sudoku in
chunks of 1,024 puzzles, Maze sweeps per checkpoint, Qwen per checkpoint.
