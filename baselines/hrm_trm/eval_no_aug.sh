#!/usr/bin/env bash
# Eval EVERY checkpoint of the no-aug runs (exact + lenient) and (re)plot.
set -euo pipefail
cd "$(dirname "$0")/../.."

uv run modal run baselines/hrm_trm/trm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-baselines/hrm_trm/trm-maze-noaug" --out baselines/hrm_trm/results/trm_noaug_sweep.json
uv run modal run baselines/hrm_trm/hrm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-baselines/hrm_trm/hrm-maze-noaug" --out baselines/hrm_trm/results/hrm_noaug_sweep.json

uv run python baselines/hrm_trm/plot_maze_sweeps.py
