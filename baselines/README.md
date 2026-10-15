# Baselines

No baseline is retrained for the camera-ready. Every baseline number comes
from an existing checkpoint, evaluated with the scripts here.

| Dir | Baseline | Benchmarks |
|---|---|---|
| [`hrm_trm/`](hrm_trm/README.md) | HRM, TRM | Maze-Hard (exact + lenient), Sudoku-Extreme (exact) |
| [`qwen_sft/`](qwen_sft/run.py) | Qwen3.5-0.8B, full SFT | Sudoku-Extreme (pass@k) |

## Evaluation protocol

- **Sudoku-Extreme:** the first N puzzles of LDT's `sudoku_test_perm`
  ordering (`lattice_diffusion.eval.indices`), so baselines and LDT see the
  same puzzles in the same order.
  - HRM/TRM: N = 10,000 by default; exact match; Clopper–Pearson 95% interval.
  - Qwen: N = 200 × 32 samples by default; unbiased pass@k.
  - Their accuracy is far from 100%, so there is no sequential stopping.
- **Maze-Hard:** the whole 1,000-puzzle test set, exact and lenient
  (any optimal path).
- **Per-puzzle rows** go to `eval/baselines/<model>/...` on the
  `lattice-diffusion-checkpoints` volume. They use the
  `lattice_diffusion/eval/records.py` field names (`idx`, `pos`, `status`,
  `solution`).

```bash
uv run python -m paper.launch launch baselines        # HRM/TRM, Sudoku + Maze
uv run python -m paper.launch launch appendix_i_llm   # Qwen SFT
```

The manifests list the exact `modal run` commands.

## Checkpoint provenance

Training times marked "derived" are step count × the measured steady-state
B200 throughput in `hrm_trm/README.md` (TRM ~1.46 s/step, HRM ~0.61 s/step).
They exclude compile and startup time.

### Maze-Hard (Table 3)

| Baseline | Checkpoint | Provenance | Training hardware / time | Exact | Lenient |
|---|---|---|---|---|---|
| TRM, B200 retrain | volume `trm-maze-aug-ckpts`: `maze-hard-repro/trm-maze-noaug/step_19530` | Our retrain, published no-aug recipe (`hrm_trm/trm_eval/modal_train.py`), global batch 768 | 1× B200; ~7.9 h to this step (derived). Run hit Modal's 24 h cap at step 58,590, before the recipe's ~65,100 steps | 85.1%† | 91.6%† |
| HRM, B200 retrain | volume `hrm-maze-aug-ckpts`: `maze-hard-repro/hrm-maze-noaug/step_23436` | Our retrain, published no-aug recipe (`hrm_trm/hrm_eval/modal_train.py`), global batch 768 | 1× B200; ~4.0 h to this step (derived) | 74.8% | 81.1% |
| HRM, B200 retrain, best lenient | same run, `step_39060` | as above | 1× B200; ~6.6 h (derived) | 72.4% | 82.2% |
| HRM, released | `sapientinc/HRM-checkpoint-maze-30x30-hard` | Authors' release | not published | 74.4% | 81.6% |
| TRM, released | `alphaXiv/trm-model-maze/maze_hard_step_32550` | alphaXiv independent reproduction (the TRM authors released no weights) | not published | 83.8%† | 90.7%† |

† Evaluated with `L_cycles=6`, but trained with `L_cycles=4`; must be
re-evaluated (see Known issues). The released checkpoint's 83.8% is close to
alphaXiv's reported 83.67%, so their number likely used the same default.

The paper's current Table 3 baselines (HRM 74.5%, TRM 85.3%; TRM "24 h,
4× L40S") are the original papers' self-reported exact-match numbers, not
from any checkpoint here.

### Sudoku-Extreme (Table 1)

| Baseline | Checkpoint | Provenance | Training hardware / time |
|---|---|---|---|
| HRM | `sapientinc/HRM-checkpoint-sudoku-extreme` (`all_config.yaml`: 40,000 epochs, global batch 768, lr 1e-4, 1k puzzles × 1,000 augmentations) | Authors' release | not published |
| TRM-MLP | `alphaXiv/trm-model-sudoku/step_32550_sudoku_epoch50k` | alphaXiv independent reproduction; model card reports 79.37% ± 0.12% vs the TRM paper's 87.4% | not published |
| TRM-attention | `alphaXiv/trm-model-sudoku/step_32550_sudoku_50k_epoch_attn_type` | alphaXiv reproduction; model card 73.66% vs paper 74.7% | not published |

The paper's current Table 1 baselines (HRM 55.0%; TRM 87.4%, "36 h, 4× L40S")
are the original papers' self-reported numbers. Neither Sudoku checkpoint
has published training hardware or time, so **B200-normalized Sudoku training
compute for HRM/TRM cannot be measured without retraining**. Only their
inference cost can be measured: the eval summaries record B200 seconds.

### Fine-tuned LLM (appendix)

Existing runs are on the `lattice-diffusion-checkpoints` volume under
`/checkpoints/followups/llm_baseline/`. Each was trained by the earlier
version of this script, Qwen3.5-0.8B full BF16 SFT for 16 epochs, with seed 0
and one checkpoint per epoch (`checkpoint-<32 × epoch>`).

| Run | Puzzles | Training (1× B200) |
|---|---|---|
| `qwen3_5_0_8b_natural_ep16_seed0` | natural (~56 blanks) | 406 s |
| `qwen3_5_0_8b_blanks1_ep16_seed0` | 1 blank | 732 s |
| `qwen3_5_0_8b_blanks2_ep16_seed0` | 2 blanks | 395 s |
| `qwen3_5_0_8b_blanks4_ep16_seed0` | 4 blanks | 407 s |
| `qwen3_5_0_8b_blanks8_ep16_seed0` | 8 blanks | 767 s |
| `qwen3_5_0_8b_blanks16_ep16_seed0` | 16 blanks | 545 s |
| `qwen3_5_0_8b_blanks32_ep16_seed0` | 32 blanks | 401 s |

These runs trained on a seed-42 1,000-puzzle subset drawn without the
`sudoku_val` holdout, so their training puzzles differ slightly from the
retrained LDT models'. New training runs (`run.py::train`) use the current
LDT pool.

## Known issues

- **Maze checkpoints are selected on the test set.** Table 3 reports each
  retrain's best checkpoint by lenient test accuracy (`maze_step: best` in
  the baselines manifest), and the caption says so. Final checkpoints for
  comparison: HRM `step_26040` 72.2% exact / 80.6% lenient; TRM `step_58590`
  77.9% / 89.6% (with the old loop count).
- **TRM Maze sweeps in `hrm_trm/results/` used the wrong loop count.** They
  ran with the fork's default `L_cycles=6`, but the checkpoints were trained
  with `L_cycles=4`. The loop count does not change parameter shapes, so this
  loads silently. The eval passes `trm_eval/cfg_maze.yaml` (`L_cycles=4`);
  rerun the sweeps before using any TRM Maze number.
- **TRM Sudoku checkpoint variant (unverified).** The mapping of
  `step_32550_sudoku_epoch50k` to TRM-MLP comes from the fork README's eval
  example and the model card's variant table. Loading it with the wrong
  config fails, because the MLP and attention weights have different shapes.
