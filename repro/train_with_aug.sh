#!/usr/bin/env bash
# Reproduce the 8x-dihedral-AUGMENTED maze-30x30-hard training runs for HRM and
# TRM (each on a single B200, 24h cap; the runs are intentionally longer than
# 24h to train as long as possible). Genuine adam-atan2 optimizer; logs stream
# to W&B (online via wandb-secret) and to <run_name>.train.log on the volume.
#
# NOTE: `modal run --detach` keeps streaming logs locally, so we background each
# launch and `wait`. The remote apps survive even if you Ctrl-C this script.
set -uo pipefail
cd "$(dirname "$0")/.."

echo ">> TRM augmented maze training (B200)"
uv run modal run --detach repro/trm_eval/modal_train.py::run_train --aug &
echo ">> HRM augmented maze training (B200)"
uv run modal run --detach repro/hrm_eval/modal_train.py::run_train --aug &
wait
