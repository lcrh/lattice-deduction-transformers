"""Pull E1 results from the Modal volume and aggregate into summary.csv.

    uv run python followups/arch_ablation/collect.py

Reads `<config>_seed<N>.eval.json` and `<config>_seed<N>.train_curve.jsonl`
for every (config, seed) in `configs.CONFIGS` from the
`lattice-diffusion-checkpoints` Modal volume under `/checkpoints/followups/e1/`,
using the Modal Volume Python API. Writes:

  results/summary.csv          one row per (config, seed) + mean/range roll-up
                               rows per config (tagged seed="mean" / "range").
  results/curves/<config>_seed<N>.train_curve.jsonl   staged locally for plots.

Missing files are skipped with a warning (a config may not have run yet). If
the volume cannot be opened at all (no auth / no network), prints a clear
message and exits non-zero.
"""

from __future__ import annotations

import csv
import json
import os
import sys

# Absolute package imports — scripts run from the repo root, so
# `followups.*` resolves via cwd (no sys.path hacks).
from followups import _common
from followups.arch_ablation import configs as C

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CURVES_DIR = os.path.join(RESULTS_DIR, "curves")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

VOLUME_NAME = C.VOLUME_NAME
VOLUME_SUBDIR = C.CKPT_SUBDIR  # "followups/e1"

# CSV columns. Params (n_params) come from the checkpoint extra and are
# re-emitted here if present in eval.json; otherwise left blank (the
# expected layers/dim from configs.py are always recorded).
COLUMNS = [
    "config", "seed", "study",
    # expected params from configs.py (source of truth for shape):
    "num_layers", "dim", "n_loops", "steps",
    "supervise", "softmax_loss_weight", "conflict_loss_weight",
    "bce_pos_mult", "bce_neg_mult",
    "approx_params",         # computed from layers/dim (relative), see below
    # eval metrics:
    "n_eval", "correct", "wrong", "timeouts", "accuracy",
    "model_calls_total", "calls_per_solve",
    "avg_rounds_solved", "avg_resets",
    "unsound_rate", "conflict_precision", "conflict_recall",
    "train_post_compile_secs",
]


def _approx_params(num_layers: int, dim: int) -> int:
    """Relative param proxy: ~ num_layers * dim^2 (transformer block scaling).

    Not exact (ignores embeddings/heads/biases); used only as a monotone
    marker-size / axis proxy for plots. Exactness comes from eval.json's
    n_params when the trainer records it.
    """
    return int(num_layers * dim * dim)


def _row_from_eval(config: str, seed: int, eval_json: dict) -> dict:
    eff = C.effective_flags(config)
    cfg_meta = C.CONFIGS[config]
    num_layers = int(eff.get("num_layers", 4))
    dim = int(eff.get("dim", 128))
    n = eval_json.get("n_eval_puzzles") or 0
    correct = eval_json.get("correct")
    wrong = eval_json.get("wrong")
    timeouts = eval_json.get("timeouts")
    calls = eval_json.get("model_calls_total")
    diag = eval_json.get("diag", {}) or {}
    wall = eval_json.get("train_wallclock", {}) or {}
    accuracy = (correct / n) if (correct is not None and n) else ""
    calls_per_solve = (calls / correct) if (calls is not None and correct) else ""
    # n_params: eval.json may not carry it; record if present else blank.
    n_params = eval_json.get("n_params", "")
    return {
        "config": config,
        "seed": seed,
        "study": cfg_meta["study"],
        "num_layers": num_layers,
        "dim": dim,
        "n_loops": eff.get("n_loops", ""),
        "steps": eff.get("steps", ""),
        "supervise": eff.get("supervise", ""),
        "softmax_loss_weight": eff.get("softmax_loss_weight", ""),
        "conflict_loss_weight": eff.get("conflict_loss_weight", ""),
        "bce_pos_mult": eff.get("bce_pos_mult", ""),
        "bce_neg_mult": eff.get("bce_neg_mult", ""),
        "approx_params": n_params if n_params != "" else _approx_params(num_layers, dim),
        "n_eval": n,
        "correct": correct if correct is not None else "",
        "wrong": wrong if wrong is not None else "",
        "timeouts": timeouts if timeouts is not None else "",
        "accuracy": accuracy,
        "model_calls_total": calls if calls is not None else "",
        "calls_per_solve": calls_per_solve,
        "avg_rounds_solved": eval_json.get("avg_rounds_solved", ""),
        "avg_resets": eval_json.get("avg_resets", ""),
        "unsound_rate": diag.get("unsound_rate", ""),
        "conflict_precision": diag.get("conflict_precision", ""),
        "conflict_recall": diag.get("conflict_recall", ""),
        "train_post_compile_secs": wall.get("post_compile_secs", ""),
    }


# Metrics that get mean / range roll-ups across seeds.
ROLLUP_METRICS = [
    "accuracy", "correct", "wrong", "timeouts",
    "model_calls_total", "calls_per_solve",
    "unsound_rate", "conflict_precision", "conflict_recall",
    "train_post_compile_secs",
]


def _rollup_rows(config: str, seed_rows: list[dict]) -> list[dict]:
    """Build seed="mean" and seed="range" roll-up rows for a config."""
    if not seed_rows:
        return []
    base = dict(seed_rows[0])  # copy shape/param columns
    mean_row = dict(base)
    range_row = dict(base)
    mean_row["seed"] = "mean"
    range_row["seed"] = "range"
    for m in ROLLUP_METRICS:
        vals = [r[m] for r in seed_rows if isinstance(r[m], (int, float))]
        if vals:
            mean_row[m] = sum(vals) / len(vals)
            range_row[m] = f"[{min(vals):.6g}, {max(vals):.6g}]"
        else:
            mean_row[m] = ""
            range_row[m] = ""
    # Non-metric columns are undefined for the range row's numeric slots;
    # leave shape columns as-is (they're seed-invariant).
    return [mean_row, range_row]


def main() -> None:
    os.makedirs(CURVES_DIR, exist_ok=True)

    try:
        vol = _common.open_volume(VOLUME_NAME)
        # Force a metadata refresh so freshly-committed files are visible.
        try:
            vol.reload()
        except Exception:  # noqa: BLE001 — reload is best-effort
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot open Modal volume {VOLUME_NAME!r}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        print("Need `modal` installed + authenticated + network access.",
              flush=True)
        sys.exit(1)

    all_rows: list[dict] = []
    n_found = n_missing = 0

    for config, cfg_meta in C.CONFIGS.items():
        seed_rows: list[dict] = []
        for seed in range(cfg_meta["n_seeds"]):
            base = f"{config}_seed{seed}"
            eval_path = f"/{VOLUME_SUBDIR}/{base}.eval.json"
            txt = _common.read_volume_text(vol, eval_path)
            if txt is None:
                n_missing += 1
                print(f"  [missing] {eval_path}", flush=True)
                continue
            try:
                eval_json = json.loads(txt)
            except json.JSONDecodeError as exc:
                print(f"  [corrupt] {eval_path}: {exc}", flush=True)
                n_missing += 1
                continue
            row = _row_from_eval(config, seed, eval_json)
            seed_rows.append(row)
            n_found += 1
            print(f"  [ok]      {base}  acc={row['accuracy']}  "
                  f"wrong={row['wrong']}", flush=True)

            # Stage the train_curve.jsonl locally for plot_all.py.
            curve_path = f"/{VOLUME_SUBDIR}/{base}.train_curve.jsonl"
            ctxt = _common.read_volume_text(vol, curve_path)
            if ctxt is not None:
                with open(os.path.join(CURVES_DIR, f"{base}.train_curve.jsonl"),
                          "w") as fh:
                    fh.write(ctxt)

        all_rows.extend(seed_rows)
        all_rows.extend(_rollup_rows(config, seed_rows))

    if n_found == 0:
        print("\nWARNING: no eval.json files found under "
              f"/{VOLUME_SUBDIR}/. Nothing collected. "
              "(Runs may not have landed yet.)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nWrote {SUMMARY_CSV}  "
          f"({n_found} runs found, {n_missing} missing).", flush=True)
    print(f"Staged train curves under {CURVES_DIR}/.", flush=True)


if __name__ == "__main__":
    main()
