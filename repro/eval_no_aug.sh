#!/usr/bin/env bash
# Eval EVERY checkpoint of the no-aug runs (exact + lenient) and (re)plot.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run modal run repro/trm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-repro/trm-maze-noaug" --out repro/results/trm_noaug_sweep.json
uv run modal run repro/hrm_eval/modal_eval.py::sweep \
    --run-dir "maze-hard-repro/hrm-maze-noaug" --out repro/results/hrm_noaug_sweep.json

uv run python repro/plot_maze_sweeps.py
