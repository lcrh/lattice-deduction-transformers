#!/usr/bin/env bash
# Same as train_with_aug.sh but WITHOUT augmentation (the published maze recipe),
# under run_name "maze_noaug". See that script's notes on backgrounding/Ctrl-C.
set -uo pipefail
cd "$(dirname "$0")/.."

echo ">> TRM no-aug maze training (B200)"
uv run modal run --detach repro/trm_eval/modal_train.py::run_train --no-aug &
echo ">> HRM no-aug maze training (B200)"
uv run modal run --detach repro/hrm_eval/modal_train.py::run_train --no-aug &
wait
