"""Assemble the normalized E8 cost table.

Columns (per model x task):
  model, task, train_gpu_h_b200, train_pflops, offline_cpu_h,
  inference_cost_per_puzzle, source

Every cell is tagged as measured / reported / TBD-measure so the provenance
stays distinguishable (README requirement). We NEVER fabricate a number: where
a same-silicon measurement doesn't exist yet, the cell is "TBD-measure" and the
source column says so.

DATA PROVENANCE
---------------
* train_pflops: analytic, from flops.py (hardware-independent). Always present.
* train_gpu_h_b200:
    - TRM/HRM Maze-Hard: MEASURED on B200 in the repro study
      (repro/README.md steady-state throughput: TRM ~1.46 s/step,
      HRM ~0.61 s/step at batch 768). GPU-h = steps * s_per_step / 3600, using
      the same step counts as flops.py (recipe point). SOURCE = measured.
    - TRM Sudoku: author-REPORTED 4x L40S x 36h = 144 L40S-h. We do NOT convert
      via a hardware factor (that's exactly the conflation E8 exists to remove);
      we mark it REPORTED and leave a B200 cell TBD-measure until
      measure_throughput.py produces a B200 s/step. If that run has landed, drop
      its steady_s_per_step into MEASURED_THROUGHPUT below and the cell flips to
      measured.
    - LDT Sudoku: MEASURED natively — experiments/sudoku/train.py records
      train_post_compile_secs (compile excluded). Once an e1/e2/e3 eval.json
      lands, read train_wallclock.post_compile_secs / (steps-1) for s/step. Until
      a committed number exists here we mark it TBD-measure (no wall-clock is
      committed in the repo yet). Its FLOPs are known regardless.
    - Sotaku: REPORTED only (no config in repo); B200-equiv TBD-measure.
* offline_cpu_h: from timed wrappers
  (`followups/cost_accounting/timed_maze_pregen.py` /
  `timed_snowflake_gen.py` print total CPU-seconds summed across workers).
  Shared `experiments/maze/pregen.py` and `experiments/snowflake/gen_data.py`
  stay uninstrumented. Sudoku ships unique solutions -> ~0 offline cost
  (stated). Numbers here are TBD-measure until those jobs run and their
  printed CPU-h is pasted into OFFLINE_CPU_H.
* inference_cost_per_puzzle: LDT has it today (sudoku/run.py writes
  forwards_unbatched / model_calls per puzzle into .eval.jsonl). TRM/HRM
  per-puzzle inference = (recurrent forward passes) * per-forward B200 cost from
  the repro checkpoints — TBD-measure until an inference-timing run lands.

Run:  uv run python followups/cost_accounting/cost_table.py
Outputs: results/cost_table.csv, results/cost_table.md, results/cost_table.tex
"""

from __future__ import annotations

import csv
import os

from followups.cost_accounting import flops

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")

# Sentinel for a not-yet-measured cell. Kept distinct from 0.0 (a real zero,
# e.g. sudoku offline cost) so "unknown" and "genuinely zero" never blur.
TBD = "TBD-measure"

# --------------------------------------------------------------------------
# MEASURED same-silicon (B200) steady-state throughput.
# repro/README.md "Steady-state training throughput (single B200, global batch
# 768)": TRM ~1.46 s/step, HRM ~0.61 s/step. These are the measured Maze-Hard
# numbers E8 reuses directly (steps * s/step, not a hardware conversion factor).
# Add TRM-sudoku here (key ("TRM","sudoku")) once measure_throughput.py lands.
# --------------------------------------------------------------------------
MEASURED_THROUGHPUT: dict[tuple[str, str], float] = {
    ("TRM", "maze-30x30-hard"): 1.46,   # s/step, B200, batch 768 (repro README)
    ("HRM", "maze-30x30-hard"): 0.61,   # s/step, B200, batch 768 (repro README)
    # ("TRM", "sudoku"): <steady_s_per_step from measure_throughput.py>,
}

# Author-REPORTED training cost on ITS OWN hardware generation (NOT converted).
# Kept as a labeled string so the reader sees the original hardware, never a
# silently-converted B200 number.
REPORTED_TRAIN: dict[tuple[str, str], str] = {
    ("TRM", "sudoku"): "144 L40S-h (4xL40S x 36h, TRM paper)",
    ("Sotaku", "sudoku"): "reported (Sotaku paper) — hardware/steps not in repo",
}

# Offline CPU-hours, summed across parallel workers, from the instrumented
# generators. Paste the printed "OFFLINE CPU COST ... CPU-h" once the gen jobs
# run. Sudoku = 0.0 (unique solutions ship with the dataset — no offline gen).
OFFLINE_CPU_H: dict[tuple[str, str], object] = {
    ("LDT", "sudoku-extreme"): 0.0,     # sudoku ships unique solutions (stated)
    ("TRM", "sudoku"): 0.0,
    ("HRM", "sudoku"): 0.0,
    ("Sotaku", "sudoku"): 0.0,
    # Maze K-path sampling (pregen.py) — TBD until the K in {1,512} timing runs:
    ("LDT", "maze-30x30-hard"): TBD,
    ("TRM", "maze-30x30-hard"): TBD,    # TRM/HRM dataset-build offline cost
    ("HRM", "maze-30x30-hard"): TBD,
    # Snowflake CVC5 gen (gen_data.py) — TBD until the per-order timing lands:
    ("LDT", "snowflake"): TBD,
}

# Per-puzzle inference cost. LDT: measured (eval.jsonl forwards_unbatched /
# model_calls). Others TBD-measure until an inference-timing run on the repro
# checkpoints lands.
INFERENCE_COST: dict[tuple[str, str], object] = {
    # ("LDT","sudoku-extreme"): <read from an eval.jsonl once committed>,
}


def _gpu_h_and_source(model: str, task: str, steps: int) -> tuple[object, str]:
    """B200-equivalent training GPU-hours and its provenance label."""
    key = (model, task)
    if key in MEASURED_THROUGHPUT:
        s_per_step = MEASURED_THROUGHPUT[key]
        gpu_h = steps * s_per_step / 3600.0
        return round(gpu_h, 3), f"measured (B200, {s_per_step:.2f} s/step x {steps} steps)"
    if key in REPORTED_TRAIN:
        # Author-reported on a DIFFERENT GPU; B200-equiv not measured -> TBD.
        return TBD, f"reported: {REPORTED_TRAIN[key]}; B200-equiv {TBD}"
    return TBD, TBD


def build_rows() -> list[dict]:
    est = {(r["model"], r["task"]): r for r in flops.all_estimates()}
    rows: list[dict] = []
    for (model, task), e in est.items():
        gpu_h, gpu_src = _gpu_h_and_source(model, task, e["steps"])
        offline = OFFLINE_CPU_H.get((model, task), TBD)
        infer = INFERENCE_COST.get((model, task), TBD)
        pflops = round(e["total_pflops"], 2) if not e["is_tbd"] else f"{e['total_pflops']:.2f} (TBD arch)"
        # Compose a single provenance string covering the whole row.
        src_bits = [f"flops={'analytic' if not e['is_tbd'] else 'TBD-arch'}",
                    f"train_gpu_h={gpu_src}"]
        rows.append({
            "model": model,
            "task": task,
            "train_gpu_h_b200": gpu_h,
            "train_pflops": pflops,
            "offline_cpu_h": offline,
            "inference_cost_per_puzzle": infer,
            "source": " | ".join(src_bits),
        })
    return rows


COLUMNS = ["model", "task", "train_gpu_h_b200", "train_pflops",
           "offline_cpu_h", "inference_cost_per_puzzle", "source"]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_markdown(rows: list[dict]) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def render_latex(rows: list[dict]) -> str:
    def esc(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    col_fmt = "l" * len(COLUMNS)
    out = [r"\begin{tabular}{" + col_fmt + "}", r"\toprule"]
    out.append(" & ".join(esc(c) for c in COLUMNS) + r" \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(" & ".join(esc(r[c]) for c in COLUMNS) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    rows = build_rows()

    csv_path = os.path.join(RESULTS, "cost_table.csv")
    md_path = os.path.join(RESULTS, "cost_table.md")
    tex_path = os.path.join(RESULTS, "cost_table.tex")

    write_csv(rows, csv_path)
    md = render_markdown(rows)
    tex = render_latex(rows)
    with open(md_path, "w") as f:
        f.write("# E8 normalized cost table\n\n")
        f.write("All GPU-hours are B200-equivalent. `measured` = same-silicon "
                "B200 measurement; `reported` = author's number on their own "
                "hardware (NOT converted); `TBD-measure` = job not yet run "
                "(never fabricated).\n\n")
        f.write(md + "\n")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")

    print(md)
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
