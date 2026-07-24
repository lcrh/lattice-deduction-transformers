"""Collect E6 eval summaries and learning curves from the Modal volume."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from followups import _common
from followups.latent_carry.configs import CKPT_SUBDIR, CONFIGS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    volume = _common.open_volume()
    rows: list[dict] = []
    missing: list[str] = []

    for config, spec in CONFIGS.items():
        for seed in range(spec["n_seeds"]):
            run = f"{config}_seed{seed}"
            eval_path = f"/{CKPT_SUBDIR}/{run}.eval.json"
            text = _common.read_volume_text(volume, eval_path)
            if text is None:
                missing.append(run)
                continue
            data = json.loads(text)
            (RESULTS / f"{run}.eval.json").write_text(
                json.dumps(data, indent=2) + "\n"
            )
            curve = _common.read_volume_text(
                volume, f"/{CKPT_SUBDIR}/{run}.train_curve.jsonl"
            )
            if curve is not None:
                (RESULTS / f"{run}.train_curve.jsonl").write_text(curve)

            n = int(data["n_evaluated_prefix"])
            diag = data.get("diag", {})
            wall = data.get("train_wallclock", {})
            rows.append({
                "config": config,
                "seed": seed,
                "carry_latent": data.get("step_cfg", {}).get("carry_latent"),
                "n": n,
                "correct": int(data["correct"]),
                "accuracy": int(data["correct"]) / max(n, 1),
                "wrong": int(data["wrong"]),
                "timeouts": int(data["timeouts"]),
                "model_calls_total": int(data["model_calls_total"]),
                "avg_puzzle_calls": float(data["avg_puzzle_calls"]),
                "avg_rounds_solved": float(data["avg_rounds_solved"]),
                "unsound_rate": float(diag.get("unsound_rate", 0.0)),
                "conflict_precision": float(diag.get("conflict_precision", 0.0)),
                "conflict_recall": float(diag.get("conflict_recall", 0.0)),
                "train_post_compile_secs": float(wall.get("post_compile_secs", 0.0)),
            })

    if rows:
        out = RESULTS / "summary.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out} ({len(rows)} rows)")
    if missing:
        print(f"Missing {len(missing)} runs: {', '.join(missing)}")


if __name__ == "__main__":
    main()
