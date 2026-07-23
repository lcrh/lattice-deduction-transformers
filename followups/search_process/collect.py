"""Pull E2 results from the Modal volume into summary.csv + per-study tables.

    uv run python followups/search_process/collect.py

Reads every config's landed eval.json (+ .eval.jsonl for the per-puzzle
forwards / conflict-depth histograms) from the `lattice-diffusion-checkpoints`
volume under `/checkpoints/followups/e2/`, using the Modal Volume Python API.
Both artifact-naming conventions are handled:

  * train configs -> `<config>_seed<N>.eval.json`
  * eval configs  -> `<config>__on__<input>_seed<N>.eval.json`

Writes:
  results/summary.csv       one row per (config, seed) + mean/range roll-ups.
  results/jsonl/<name>.eval.jsonl   staged per-puzzle rows (for the histograms).

The KEY-COST metric row (README): every row reports solve rate, puzzle_calls
(batched cost), the sequential-forwards estimate (p50/p90 from the jsonl), and
resets. S3 rows additionally carry unsound_negation_rate.

Missing files are skipped with a warning. If the volume cannot be opened at all,
prints a clear message and exits non-zero (matches E1/E3).
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys

from followups import _common
from followups.search_process import configs as C

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
JSONL_DIR = os.path.join(RESULTS_DIR, "jsonl")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

VOLUME_NAME = C.VOLUME_NAME
VOLUME_SUBDIR = C.CKPT_SUBDIR  # "followups/e2"

COLUMNS = [
    "config", "seed", "study", "kind", "input",
    "cell_policy", "digit_policy", "backtrack",
    # eval metrics:
    "n_eval", "correct", "wrong", "timeouts", "accuracy",
    "model_calls_total", "calls_per_solve",
    "avg_resets", "avg_puzzle_calls",
    # sequential-cost (from jsonl forwards_seq):
    "seq_forwards_p50", "seq_forwards_p90",
    # soundness / backtracking diagnostics:
    "unsound_rate", "conflict_precision", "conflict_recall",
    "n_negations", "unsound_negation_rate", "conflict_depth_mean",
]

# Metrics rolled up (mean / range) across seeds.
ROLLUP_METRICS = [
    "accuracy", "correct", "wrong", "timeouts",
    "model_calls_total", "calls_per_solve", "avg_resets", "avg_puzzle_calls",
    "seq_forwards_p50", "seq_forwards_p90",
    "unsound_rate", "conflict_precision", "conflict_recall",
    "unsound_negation_rate", "conflict_depth_mean",
]


def _policy_of(config: str) -> tuple[str, str, str]:
    """(cell_policy, digit_policy, backtrack) as configured for this config."""
    cfg = C.CONFIGS[config]
    if cfg["kind"] == "eval":
        eff = C.effective_eval_flags(config)
        return (str(eff.get("cell_policy", "uniform")),
                str(eff.get("digit_policy", "softmax")),
                str(eff.get("backtrack", "root")))
    ov = cfg["overrides"]
    return (str(ov.get("cell_policy", "uniform")),
            str(ov.get("digit_policy", "softmax")),
            str(ov.get("backtrack", "root")))


def _percentiles(values: list[float]) -> tuple[float, float]:
    """(p50, p90) of a list; ('', '') if empty."""
    if not values:
        return "", ""
    s = sorted(values)
    def pct(p):
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return pct(0.50), pct(0.90)


def _row_from_eval(config: str, seed: int, ej: dict, jsonl_rows: list[dict]) -> dict:
    cfg = C.CONFIGS[config]
    cp, dp, bt = _policy_of(config)
    # eval.json layout differs between run.py (flat) and eval_only.py (nested).
    summary = ej.get("summary", ej)
    diag = ej.get("diag", {}) or {}
    n = ej.get("n_eval") or ej.get("n_eval_puzzles") or 0
    correct = summary.get("correct", ej.get("correct"))
    wrong = summary.get("wrong", ej.get("wrong"))
    timeouts = summary.get("timeouts", ej.get("timeouts"))
    calls = summary.get("total_calls", ej.get("model_calls_total"))
    avg_resets = summary.get("avg_resets", ej.get("avg_resets", ""))
    avg_puzzle_calls = summary.get("avg_puzzle_calls", "")
    accuracy = (correct / n) if (correct is not None and n) else ""
    calls_per_solve = (calls / correct) if (calls and correct) else ""
    # sequential-forwards percentiles from the per-puzzle jsonl.
    seq_vals = [r["forwards_seq"] for r in jsonl_rows
                if r.get("kind") == "puzzle" and isinstance(r.get("forwards_seq"), (int, float))
                and r["forwards_seq"] >= 0]
    seq_p50, seq_p90 = _percentiles([float(v) for v in seq_vals])
    return {
        "config": config, "seed": seed, "study": cfg["study"],
        "kind": cfg["kind"], "input": cfg.get("input", ""),
        "cell_policy": cp, "digit_policy": dp, "backtrack": bt,
        "n_eval": n,
        "correct": correct if correct is not None else "",
        "wrong": wrong if wrong is not None else "",
        "timeouts": timeouts if timeouts is not None else "",
        "accuracy": accuracy,
        "model_calls_total": calls if calls is not None else "",
        "calls_per_solve": calls_per_solve,
        "avg_resets": avg_resets,
        "avg_puzzle_calls": avg_puzzle_calls,
        "seq_forwards_p50": seq_p50,
        "seq_forwards_p90": seq_p90,
        "unsound_rate": diag.get("unsound_rate", ""),
        "conflict_precision": diag.get("conflict_precision", ""),
        "conflict_recall": diag.get("conflict_recall", ""),
        "n_negations": diag.get("n_negations", ""),
        "unsound_negation_rate": diag.get("unsound_negation_rate", ""),
        "conflict_depth_mean": diag.get("conflict_depth_mean", ""),
    }


def _rollup_rows(config: str, seed_rows: list[dict]) -> list[dict]:
    if not seed_rows:
        return []
    base = dict(seed_rows[0])
    mean_row = dict(base); range_row = dict(base)
    mean_row["seed"] = "mean"; range_row["seed"] = "range"
    for m in ROLLUP_METRICS:
        vals = [r[m] for r in seed_rows if isinstance(r[m], (int, float))]
        if vals:
            mean_row[m] = sum(vals) / len(vals)
            range_row[m] = f"[{min(vals):.6g}, {max(vals):.6g}]"
        else:
            mean_row[m] = ""; range_row[m] = ""
    return [mean_row, range_row]


def main() -> None:
    os.makedirs(JSONL_DIR, exist_ok=True)
    try:
        vol = _common.open_volume(VOLUME_NAME)
        try:
            vol.reload()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot open Modal volume {VOLUME_NAME!r}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        print("Need `modal` installed + authenticated + network access.", flush=True)
        sys.exit(1)

    all_rows: list[dict] = []
    n_found = n_missing = 0

    for config, cfg_meta in C.CONFIGS.items():
        seed_rows: list[dict] = []
        for seed in range(cfg_meta["n_seeds"]):
            base = C.output_name(config, seed)
            eval_path = f"/{VOLUME_SUBDIR}/{base}.eval.json"
            txt = _common.read_volume_text(vol, eval_path)
            if txt is None:
                n_missing += 1
                print(f"  [missing] {eval_path}", flush=True)
                continue
            try:
                ej = json.loads(txt)
            except json.JSONDecodeError as exc:
                print(f"  [corrupt] {eval_path}: {exc}", flush=True)
                n_missing += 1
                continue
            # Stage the per-puzzle jsonl (for histograms + seq percentiles).
            jsonl_rows: list[dict] = []
            jtxt = _common.read_volume_text(vol, f"/{VOLUME_SUBDIR}/{base}.eval.jsonl")
            if jtxt is not None:
                with open(os.path.join(JSONL_DIR, f"{base}.eval.jsonl"), "w") as fh:
                    fh.write(jtxt)
                for line in jtxt.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            jsonl_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            row = _row_from_eval(config, seed, ej, jsonl_rows)
            seed_rows.append(row)
            n_found += 1
            print(f"  [ok]      {base}  acc={row['accuracy']}  "
                  f"bt={row['backtrack']}", flush=True)
        all_rows.extend(seed_rows)
        all_rows.extend(_rollup_rows(config, seed_rows))

    if n_found == 0:
        print(f"\nWARNING: no eval.json under /{VOLUME_SUBDIR}/. Nothing "
              "collected (runs may not have landed).", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nWrote {SUMMARY_CSV}  ({n_found} runs found, {n_missing} missing).",
          flush=True)
    print(f"Staged per-puzzle jsonl under {JSONL_DIR}/.", flush=True)


if __name__ == "__main__":
    main()
