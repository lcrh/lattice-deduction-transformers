#!/usr/bin/env bash
# Eval EVERY checkpoint of the augmented runs (exact + lenient/any-optimal-path)
# and plot accuracy-over-training. Assumes train_with_aug.sh has finished and
# its checkpoints are committed to the volumes.
set -euo pipefail
cd "$(dirname "$0")/../.."

uv run modal run baselines/hrm_trm/trm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-repro/trm-maze-aug" --out baselines/hrm_trm/results/trm_aug_sweep.json
uv run modal run baselines/hrm_trm/hrm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-repro/hrm-maze-aug" --out baselines/hrm_trm/results/hrm_aug_sweep.json

uv run python baselines/hrm_trm/plot_maze_sweeps.py
