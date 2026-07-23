"""E4 (ood_snowflake) collect — pull eval results + aggregate PER ORDER.

Reads the `<config>_seed<N>.eval.json` artifacts that E4 training writes to the
`lattice-diffusion-checkpoints` Modal volume under `/checkpoints/followups/e4/`,
and aggregates the per-order breakdown the README asks for:

  - accuracy per test order          = correct / (correct + wrong + timeout)
  - soundness per test order         = wrong count (kept SEPARATE from timeouts)
  - timeout count per test order
  - calls/solve per test order       = calls / correct  (search cost)
  - pooled (all-order) accuracy per config

The interesting signal is accuracy vs. distance from the training range, plus
whether far-OOD failures show up as ABSTENTIONS (timeouts) or UNSOUND wrong
answers (README: "keep the two outcomes separate").

Consumes the `per_order` key written by `experiments/snowflake/run.py`:

    "per_order": { "<order>": {"correct": int, "wrong": int,
                               "timeout": int, "calls": int, "n": int}, ... }

(`calls` is the summed per-puzzle model calls for that order; `n` the number of
SAT eval puzzles at that order.)

Outputs (created under followups/ood_snowflake/results/):
  - per_order.csv : one row per (config, train_orders, test_order, seed) with
    raw counts + accuracy, PLUS roll-up rows (seed="mean", seed="min",
    seed="max") aggregating across the available seeds for that
    (config, test_order).
  - summary.csv   : one row per config with pooled (across orders & seeds)
    accuracy / soundness / timeout-rate + calls/solve.

Usage:
    uv run python followups/ood_snowflake/collect.py
    uv run python followups/ood_snowflake/collect.py --subdir followups/e4

Resilience: if the whole volume read fails (no modal / no auth), prints a clear
message and exits non-zero. If only some artifacts are missing, it warns and
proceeds on the partial data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Absolute package imports rooted at the repo root. Scripts are run from the
# repo root (`uv run python followups/ood_snowflake/collect.py`), so cwd is on
# sys.path and `followups` resolves as a package (every dir has __init__.py).
from followups import _common  # shared volume helpers
from followups.ood_snowflake import configs as E4  # the run matrix

_HERE = Path(__file__).resolve().parent

VOLUME_NAME = E4.VOLUME_NAME
RESULTS_DIR = _HERE / "results"


def _read_eval_jsons(subdir: str) -> tuple[dict[str, dict], list[str]]:
    """Return ({<config>_seed<N>: parsed_json}, [warnings]) for `/<subdir>/`.

    Lists the volume dir, reads every `*.eval.json` via the shared helpers
    (`_common.open_volume` + `_common.read_volume_text`), and parses each.
    Individual read/parse failures are collected as warnings, not fatal; a
    failure to open/list the volume propagates (fatal — handled by main()).
    """
    import json
    vol = _common.open_volume(VOLUME_NAME)
    try:
        vol.reload()
    except Exception:  # noqa: BLE001 — best-effort metadata refresh
        pass
    root = "/" + subdir.strip("/")
    suffix = ".eval.json"
    names = [
        entry.path.rsplit("/", 1)[-1]
        for entry in vol.iterdir(root)
        if entry.path.endswith(suffix)
    ]
    parsed: dict[str, dict] = {}
    warnings: list[str] = []
    for base in sorted(names):
        vol_path = f"{root}/{base}"
        try:
            text = _common.read_volume_text(vol, vol_path)
            if text is None:
                warnings.append(f"{vol_path} vanished between list and read")
                continue
            obj = json.loads(text)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"could not read/parse {vol_path}: {e}")
            continue
        parsed[base[: -len(suffix)]] = obj
    return parsed, warnings


# --------------------------------------------------------------------------
# Aggregation.
# --------------------------------------------------------------------------

def _accuracy(correct: int, wrong: int, timeout: int) -> float:
    denom = correct + wrong + timeout
    return (correct / denom) if denom > 0 else 0.0


def _calls_per_solve(calls: int, correct: int) -> float:
    return (calls / correct) if correct > 0 else 0.0


def collect(parsed: dict[str, dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Build per-order rows, summary rows, and notes from parsed eval.jsons.

    `parsed` maps `<config>_seed<N>` -> eval.json dict. Only configs present in
    E4.CONFIGS are considered; unexpected keys are noted and skipped.
    """
    notes: list[str] = []

    # per_seed[(config, order)][seed] = {correct, wrong, timeout, calls, n}
    per_seed: dict[tuple[str, int], dict[int, dict]] = {}
    # pooled_by_config[config][seed] = summed-over-orders counts
    pooled: dict[str, dict[int, dict]] = {}

    for name in E4.CONFIGS:
        for seed in range(E4.CONFIGS[name]["n_seeds"]):
            key = f"{name}_seed{seed}"
            if key not in parsed:
                notes.append(f"missing artifact: {key}.eval.json (skipped)")
                continue
            obj = parsed[key]
            po = obj.get("per_order")
            if not isinstance(po, dict):
                notes.append(f"{key}: no 'per_order' key in eval.json (skipped)")
                continue
            pooled.setdefault(name, {})[seed] = {
                "correct": 0, "wrong": 0, "timeout": 0, "calls": 0, "n": 0}
            for order_str, b in po.items():
                try:
                    order = int(order_str)
                except (TypeError, ValueError):
                    notes.append(f"{key}: non-integer order {order_str!r} (skipped)")
                    continue
                rec = {
                    "correct": int(b.get("correct", 0)),
                    "wrong": int(b.get("wrong", 0)),
                    "timeout": int(b.get("timeout", 0)),
                    "calls": int(b.get("calls", 0)),
                    "n": int(b.get("n", b.get("correct", 0) + b.get("wrong", 0)
                                    + b.get("timeout", 0))),
                }
                per_seed.setdefault((name, order), {})[seed] = rec
                p = pooled[name][seed]
                for k in ("correct", "wrong", "timeout", "calls", "n"):
                    p[k] += rec[k]

    # ---- per_order.csv rows ----
    per_order_rows: list[dict] = []
    for name in E4.CONFIGS:
        train_orders = E4.CONFIGS[name]["train_orders"]
        boundary = E4.max_train_order(name)
        orders = sorted({o for (c, o) in per_seed if c == name})
        for order in orders:
            seed_recs = per_seed[(name, order)]
            accs = []
            for seed in sorted(seed_recs):
                r = seed_recs[seed]
                acc = _accuracy(r["correct"], r["wrong"], r["timeout"])
                accs.append(acc)
                per_order_rows.append({
                    "config": name,
                    "train_orders": train_orders,
                    "train_max_order": boundary,
                    "test_order": order,
                    "is_ood": int(order > boundary),
                    "seed": str(seed),
                    "correct": r["correct"],
                    "wrong": r["wrong"],
                    "timeout": r["timeout"],
                    "calls": r["calls"],
                    "n": r["n"],
                    "accuracy": round(acc, 6),
                    "calls_per_solve": round(
                        _calls_per_solve(r["calls"], r["correct"]), 4),
                })
            # roll-up across seeds for this (config, order)
            if accs:
                mean_acc = sum(accs) / len(accs)
                agg = {"correct": 0, "wrong": 0, "timeout": 0, "calls": 0, "n": 0}
                for seed in seed_recs:
                    for k in agg:
                        agg[k] += seed_recs[seed][k]
                for seed_label, acc_val in (
                    ("mean", round(mean_acc, 6)),
                    ("min", round(min(accs), 6)),
                    ("max", round(max(accs), 6)),
                ):
                    per_order_rows.append({
                        "config": name,
                        "train_orders": train_orders,
                        "train_max_order": boundary,
                        "test_order": order,
                        "is_ood": int(order > boundary),
                        "seed": seed_label,
                        # counts are the pooled-across-seed totals; only
                        # meaningful on the "mean" roll-up, repeated for min/max.
                        "correct": agg["correct"],
                        "wrong": agg["wrong"],
                        "timeout": agg["timeout"],
                        "calls": agg["calls"],
                        "n": agg["n"],
                        "accuracy": acc_val,
                        "calls_per_solve": round(
                            _calls_per_solve(agg["calls"], agg["correct"]), 4),
                    })

    # ---- summary.csv rows (pooled over orders AND seeds per config) ----
    summary_rows: list[dict] = []
    for name in E4.CONFIGS:
        seed_map = pooled.get(name, {})
        n_seeds_done = len(seed_map)
        agg = {"correct": 0, "wrong": 0, "timeout": 0, "calls": 0, "n": 0}
        for seed in seed_map:
            for k in agg:
                agg[k] += seed_map[seed][k]
        total = agg["correct"] + agg["wrong"] + agg["timeout"]
        summary_rows.append({
            "config": name,
            "train_orders": E4.CONFIGS[name]["train_orders"],
            "eval_orders": E4.CONFIGS[name]["eval_orders"],
            "train_max_order": E4.max_train_order(name),
            "n_seeds_done": n_seeds_done,
            "n_eval": total,
            "correct": agg["correct"],
            "wrong": agg["wrong"],
            "timeout": agg["timeout"],
            "pooled_accuracy": round(_accuracy(
                agg["correct"], agg["wrong"], agg["timeout"]), 6),
            "wrong_rate": round((agg["wrong"] / total) if total else 0.0, 6),
            "timeout_rate": round((agg["timeout"] / total) if total else 0.0, 6),
            "calls_per_solve": round(
                _calls_per_solve(agg["calls"], agg["correct"]), 4),
        })

    return per_order_rows, summary_rows, notes


# --------------------------------------------------------------------------
# CSV writing.
# --------------------------------------------------------------------------

PER_ORDER_COLUMNS = [
    "config", "train_orders", "train_max_order", "test_order", "is_ood",
    "seed", "correct", "wrong", "timeout", "calls", "n",
    "accuracy", "calls_per_solve",
]
SUMMARY_COLUMNS = [
    "config", "train_orders", "eval_orders", "train_max_order",
    "n_seeds_done", "n_eval", "correct", "wrong", "timeout",
    "pooled_accuracy", "wrong_rate", "timeout_rate", "calls_per_solve",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdir", default=E4.CKPT_SUBDIR,
                    help=f"volume subdir to read (default {E4.CKPT_SUBDIR!r}).")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="local dir for per_order.csv / summary.csv.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Whole-volume read: fatal on failure.
    try:
        parsed, read_warnings = _read_eval_jsons(args.subdir)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not read the {VOLUME_NAME!r} volume: {e}",
              file=sys.stderr)
        sys.exit(1)

    for w in read_warnings:
        print(f"  warning: {w}", file=sys.stderr)

    if not parsed:
        print(f"ERROR: no *.eval.json artifacts found under /{args.subdir} on "
              f"the {VOLUME_NAME!r} volume. Launch the E4 sweep first "
              "(`configs.py list`).", file=sys.stderr)
        sys.exit(1)

    per_order_rows, summary_rows, notes = collect(parsed)

    for n in notes:
        print(f"  note: {n}", file=sys.stderr)

    per_order_path = out_dir / "per_order.csv"
    summary_path = out_dir / "summary.csv"
    _write_csv(per_order_path, PER_ORDER_COLUMNS, per_order_rows)
    _write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)

    n_configs_with_data = len({r["config"] for r in per_order_rows})
    print(f"Wrote {per_order_path} ({len(per_order_rows)} rows).")
    print(f"Wrote {summary_path} ({len(summary_rows)} rows, "
          f"{n_configs_with_data} configs with data).")
    if notes:
        print(f"({len(notes)} note(s) above — some artifacts missing/partial.)")


if __name__ == "__main__":
    main()
