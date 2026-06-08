#!/usr/bin/env bash
# Eval EVERY checkpoint of the augmented runs (exact + lenient/any-optimal-path)
# and plot accuracy-over-training. Assumes train_with_aug.sh has finished and
# its checkpoints are committed to the volumes.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run modal run repro/trm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-repro/trm-maze-aug" --out repro/results/trm_aug_sweep.json
uv run modal run repro/hrm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-repro/hrm-maze-aug" --out repro/results/hrm_aug_sweep.json

uv run python repro/plot_maze_sweeps.py
