"""Shared helpers for the table and figure scripts: read collected results,
aggregate over seeds, paired comparisons from per-puzzle rows, LaTeX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from lattice_diffusion.eval.stats import PairedDifference, paired_difference

from paper.manifest import REPO

RESULTS = REPO / "paper" / "results"
OUT = REPO / "paper" / "out"

_NUMERIC = {"seed", "eval_seed", "n", "correct", "wrong", "timeout", "accuracy", "acc_lo",
            "acc_hi", "wrong_rate_upper", "failure_rate", "calls_mean", "calls_p50",
            "calls_p75", "calls_p90", "calls_p95", "calls_correct_mean", "unsound_rate",
            "gpu_seconds", "startup_seconds", "wall_seconds", "n_evaluated", "train_seconds",
            "model_dim", "model_layers", "model_loops", "model_heads", "model_seq_len",
            "model_ffn_mult", "model_channels", "train_steps", "train_batch", "solve_chains",
            "solve_max_rounds", "x"}

EVAL_RULES = {
    "fixed_n": ("the first {n} puzzles of a fixed random ordering of the test split, "
                "the same puzzles for every row and seed"),
    "exhausted": "every puzzle of the evaluated list ({n} per seed)",
    "failures": ("a fixed random ordering of the test split, evaluated until 100 failures "
                 "(n = {n} per seed)"),
    "half_width": ("a fixed random ordering of the test split, evaluated until the 95% "
                   "interval on accuracy is within 0.1 points (n = {n} per seed)"),
}


def parse_cell(key: str, value: str):
    if value == "" or value is None:
        return None
    if key in _NUMERIC:
        try:
            f = float(value)
        except ValueError:
            return value
        return int(f) if f.is_integer() and key not in ("accuracy", "x") else f
    if value in ("True", "False"):
        return value == "True"
    return value


def read_results(manifest: str, results_dir: Path = RESULTS) -> list[dict]:
    path = results_dir / f"{manifest}.csv"
    with path.open() as fh:
        return [{k: parse_cell(k, v) for k, v in row.items()} for row in csv.DictReader(fh)]


def by_entry(records: list[dict]) -> dict[str, list[dict]]:
    """Records grouped by entry, in first-appearance (manifest) order."""
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["entry"], []).append(r)
    return out


def done(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("status") == "done" and r.get("n") is not None]


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return math.nan, math.nan
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, math.nan
    return mu, math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def load_rows(record: dict, root: Path = REPO) -> list[dict]:
    path = record.get("rows_path")
    if not path:
        raise FileNotFoundError(f"{record['key']}: no rows (collect with --rows)")
    with (root / path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def correct_by_pos(rows: list[dict]) -> list[bool]:
    return [r["status"] == "correct" for r in sorted(rows, key=lambda r: r["pos"])]


def paired_vs(reference: list[dict], arm: list[dict], root: Path = REPO) -> PairedDifference:
    """Arm minus reference, paired puzzle by puzzle within each seed (same
    seed, same positions), pooled over the seeds both have."""
    ref_by_seed = {r["seed"]: r for r in reference}
    a, b = [], []
    for rec in arm:
        ref = ref_by_seed.get(rec["seed"])
        if ref is None:
            continue
        ca, cb = correct_by_pos(load_rows(rec, root)), correct_by_pos(load_rows(ref, root))
        n = min(len(ca), len(cb))
        a += ca[:n]
        b += cb[:n]
    return paired_difference(a, b)


def eval_rule(records: list[dict]) -> str:
    """Caption text for how a set of runs was evaluated."""
    reasons = {r["stop_reason"] for r in records if r.get("stop_reason")}
    ns = sorted({r["n"] for r in records if r.get("n") is not None})
    n_text = f"{ns[0]:,}" if len(ns) == 1 else f"{ns[0]:,}--{ns[-1]:,}" if ns else "?"
    if len(reasons) == 1:
        return "Evaluated on " + EVAL_RULES.get(next(iter(reasons)), "{n} puzzles").format(n=n_text)
    return f"Evaluated sequentially on a fixed random ordering of the test split (n = {n_text})"


def pct(x: float | None, digits: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


def pm(mu: float, sd: float, digits: int = 1) -> str:
    if math.isnan(mu):
        return "--"
    if math.isnan(sd):
        return f"{100 * mu:.{digits}f}"
    return f"{100 * mu:.{digits}f} $\\pm$ {100 * sd:.{digits}f}"


def delta_text(d: PairedDifference) -> str:
    if d.n == 0:
        return "--"
    return f"{100 * d.delta:+.1f} [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]"


def num(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:,.{digits}f}" if isinstance(x, float) else f"{x:,}"


def escape(text: str) -> str:
    return (str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#"))


@dataclass
class Table:
    columns: list[str]                  # header cells (LaTeX)
    rows: list[list[str]]               # body cells (LaTeX); a list of one str = midrule label
    caption: str
    label: str
    align: str | None = None

    def to_latex(self) -> str:
        align = self.align or "l" + "r" * (len(self.columns) - 1)
        lines = ["\\begin{table}[t]", "\\centering", "\\small",
                 f"\\caption{{{self.caption}}}", f"\\label{{{self.label}}}",
                 f"\\begin{{tabular}}{{{align}}}", "\\toprule",
                 " & ".join(self.columns) + " \\\\", "\\midrule"]
        for row in self.rows:
            if len(row) == 1:
                lines += ["\\midrule",
                          f"\\multicolumn{{{len(self.columns)}}}{{l}}{{\\emph{{{row[0]}}}}} \\\\"]
            else:
                lines.append(" & ".join(row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        return "\n".join(lines)


def write_output(name: str, content: str | bytes, out_dir: Path = OUT,
                 copy_to: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, copy_to / name)
    return path


def cli(description: str, default_manifest: str | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    if default_manifest is None:
        ap.add_argument("manifest")
    else:
        ap.add_argument("--manifest", default=default_manifest)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--copy-to", type=Path, default=None,
                    help="also copy outputs here (e.g. ../latticeDiffusion/figs)")
    return ap.parse_args()
