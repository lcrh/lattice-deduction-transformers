"""Pull E3 results from the Modal volume and aggregate into summary CSVs.

    uv run python followups/deduction_operator/collect.py

Reads, for every (config, seed) in `configs.CONFIGS`, the eval-only artifact
`<evalconfig>__on__<input>_seed<N>.eval.json` (eval_only.py's schema:
`solver_config` / `summary` / `diag`) from the `lattice-diffusion-checkpoints`
volume under `/checkpoints/followups/e3/`. Also stages the profiler CSV
(`profile_iters.csv`) locally for plot_all.py's O2 figure.

Writes:
  results/summary.csv       one row per (config, seed) + mean/range roll-ups.
  results/o3_table.csv       the O3 passes-per-round comparison table.
  results/profile_iters.csv  staged copy of the O2 profiler output (if present).

Missing files are skipped with a warning (a config may not have run yet). If
the volume cannot be opened at all (no auth / no network), prints a clear
message and exits non-zero.
"""

from __future__ import annotations

import csv
import json
import os
import sys

from followups import _common
from followups.deduction_operator import configs as C

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
O3_TABLE_CSV = os.path.join(RESULTS_DIR, "o3_table.csv")
PROFILE_CSV = os.path.join(RESULTS_DIR, "profile_iters.csv")

VOLUME_NAME = C.VOLUME_NAME
VOLUME_SUBDIR = C.CKPT_SUBDIR  # "followups/e3"

COLUMNS = [
    "config", "seed", "study", "input",
    # operating point (from configs.py, the source of truth):
    "effective_n_loops", "eval_n_loops_flag", "deduce_passes",
    "threshold", "cls_threshold", "augment",
    # eval metrics (from eval.json):
    "n_eval", "correct", "wrong", "timeouts", "accuracy",
    "total_calls", "calls_per_solve", "avg_resets",
    "unsound_rate", "conflict_precision", "conflict_recall",
]


def _row_from_eval(config: str, seed: int, ev: dict) -> dict:
    eff = C.effective_flags(config)
    meta = C.CONFIGS[config]
    sc = ev.get("solver_config", {}) or {}
    summ = ev.get("summary", {}) or {}
    diag = ev.get("diag", {}) or {}
    n = ev.get("n_eval") or 0
    correct = summ.get("correct")
    wrong = summ.get("wrong")
    timeouts = summ.get("timeouts")
    calls = summ.get("total_calls")
    accuracy = (correct / n) if (correct is not None and n) else ""
    calls_per_solve = (calls / correct) if (calls and correct) else ""
    return {
        "config": config,
        "seed": seed,
        "study": meta["study"],
        "input": meta["input"],
        # Prefer the value the eval actually recorded; fall back to configs.
        "effective_n_loops": sc.get("effective_n_loops",
                                    C.effective_n_loops(config)),
        "eval_n_loops_flag": eff.get("eval_n_loops", 0),
        "deduce_passes": sc.get("deduce_passes", eff.get("deduce_passes", 1)),
        "threshold": sc.get("threshold", eff.get("threshold", "")),
        "cls_threshold": sc.get("cls_threshold", eff.get("cls_threshold", "")),
        "augment": sc.get("augment", eff.get("augment", "")),
        "n_eval": n,
        "correct": correct if correct is not None else "",
        "wrong": wrong if wrong is not None else "",
        "timeouts": timeouts if timeouts is not None else "",
        "accuracy": accuracy,
        "total_calls": calls if calls is not None else "",
        "calls_per_solve": calls_per_solve,
        "avg_resets": summ.get("avg_resets", ""),
        "unsound_rate": diag.get("unsound_rate", ""),
        "conflict_precision": diag.get("conflict_precision", ""),
        "conflict_recall": diag.get("conflict_recall", ""),
    }


ROLLUP_METRICS = [
    "accuracy", "correct", "wrong", "timeouts",
    "total_calls", "calls_per_solve", "avg_resets",
    "unsound_rate", "conflict_precision", "conflict_recall",
]


def _rollup_rows(config: str, seed_rows: list[dict]) -> list[dict]:
    if not seed_rows:
        return []
    base = dict(seed_rows[0])
    mean_row = dict(base); mean_row["seed"] = "mean"
    range_row = dict(base); range_row["seed"] = "range"
    for m in ROLLUP_METRICS:
        vals = [r[m] for r in seed_rows if isinstance(r[m], (int, float))]
        if vals:
            mean_row[m] = sum(vals) / len(vals)
            range_row[m] = f"[{min(vals):.6g}, {max(vals):.6g}]"
        else:
            mean_row[m] = ""; range_row[m] = ""
    return [mean_row, range_row]


# The O3 table row order (config -> passes-per-round label).
O3_ROWS = [
    ("o1_scale_L16", "baseline (1 pass, native L16)"),
    ("o3_d2", "2 passes"),
    ("o3_d4", "4 passes"),
    ("o3_fix", "fixpoint (cap 16)"),
    ("o3_d4_noaug", "4 passes, no aug (control)"),
]


def _write_o3_table(by_config: dict[str, list[dict]]) -> None:
    cols = ["config", "passes_label", "accuracy", "wrong",
            "total_calls", "calls_per_solve", "avg_resets", "unsound_rate"]
    with open(O3_TABLE_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for cfg_name, label in O3_ROWS:
            rows = by_config.get(cfg_name, [])
            if not rows:
                w.writerow({"config": cfg_name, "passes_label": label})
                continue

            def _mean(col):
                vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
                return (sum(vals) / len(vals)) if vals else ""
            w.writerow({
                "config": cfg_name, "passes_label": label,
                "accuracy": _mean("accuracy"), "wrong": _mean("wrong"),
                "total_calls": _mean("total_calls"),
                "calls_per_solve": _mean("calls_per_solve"),
                "avg_resets": _mean("avg_resets"),
                "unsound_rate": _mean("unsound_rate"),
            })
    print(f"Wrote {O3_TABLE_CSV}", flush=True)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        vol = _common.open_volume(VOLUME_NAME)
        try:
            vol.reload()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot open Modal volume {VOLUME_NAME!r}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        print("Need `modal` installed + authenticated + network access.",
              flush=True)
        sys.exit(1)

    all_rows: list[dict] = []
    by_config: dict[str, list[dict]] = {}
    n_found = n_missing = 0

    for config, meta in C.CONFIGS.items():
        seed_rows: list[dict] = []
        for seed in range(meta["n_seeds"]):
            out = C.output_name(config, seed)
            eval_path = f"/{VOLUME_SUBDIR}/{out}.eval.json"
            txt = _common.read_volume_text(vol, eval_path)
            if txt is None:
                n_missing += 1
                print(f"  [missing] {eval_path}", flush=True)
                continue
            try:
                ev = json.loads(txt)
            except json.JSONDecodeError as exc:
                print(f"  [corrupt] {eval_path}: {exc}", flush=True)
                n_missing += 1
                continue
            row = _row_from_eval(config, seed, ev)
            seed_rows.append(row)
            n_found += 1
            print(f"  [ok]      {out}  acc={row['accuracy']}  "
                  f"wrong={row['wrong']}", flush=True)
        all_rows.extend(seed_rows)
        all_rows.extend(_rollup_rows(config, seed_rows))
        by_config[config] = seed_rows

    # Stage the profiler CSV locally (O2).
    prof = _common.read_volume_text(vol, f"/{VOLUME_SUBDIR}/profile_iters.csv")
    if prof is not None:
        with open(PROFILE_CSV, "w") as fh:
            fh.write(prof)
        print(f"Staged O2 profiler CSV -> {PROFILE_CSV}", flush=True)
    else:
        print("  [missing] profile_iters.csv (O2 profiler not run yet)",
              flush=True)

    if n_found == 0:
        print("\nWARNING: no eval.json files found under "
              f"/{VOLUME_SUBDIR}/. Nothing collected. "
              "(Runs may not have landed yet.)", flush=True)

    with open(SUMMARY_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nWrote {SUMMARY_CSV}  ({n_found} runs found, {n_missing} missing).",
          flush=True)

    _write_o3_table(by_config)


if __name__ == "__main__":
    main()
