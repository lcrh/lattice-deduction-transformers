"""Shared helpers for the table and figure scripts."""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from ldt.eval.stats import PairedDifference, paired_difference

from paper.manifest import REPO
from paper.results import RESULTS, read_results

__all__ = ["RESULTS", "read_results"]

OUT = REPO / "paper" / "out"

EVAL_RULES = {
    "fixed_n": ("the first {n} puzzles of a fixed random ordering of the test split, "
                "the same puzzles for every row and seed"),
    "exhausted": "every puzzle of the evaluated list ({n} per seed)",
    "failures": ("a fixed random ordering of the test split, evaluated until 100 failures "
                 "(n = {n} per seed)"),
    "half_width": ("a fixed random ordering of the test split, evaluated until the 95% "
                   "interval on accuracy is within 0.1 points (n = {n} per seed)"),
    "n_max": "a fixed random ordering of the puzzles, evaluated up to a cap (n = {n} per seed)",
}


def by_entry(records: list[dict]) -> dict[str, list[dict]]:
    """Records grouped by entry, in first-appearance (manifest) order."""
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["entry"], []).append(r)
    return out


def done(records: list[dict]) -> list[dict]:
    """Records with a finished eval summary."""
    return [r for r in records if r.get("status") == "done" and r.get("n") is not None]


def seeds_cell(recs: list[dict]) -> str:
    """Finished runs of one entry: "3", or "2/3" with a printed warning when some are missing."""
    n_done = len(done(recs))
    if n_done < len(recs):
        print(f"warning: {recs[0]['entry']}: {n_done} of {len(recs)} runs finished")
        return f"{n_done}/{len(recs)}"
    return str(n_done)


def seeds_text(records: list[dict]) -> str:
    """Caption text for the number of finished seeds per entry, e.g. "3" or "2--3"."""
    return n_range([len(v) for v in by_entry(done(records)).values()])


def mean(values: list[float]) -> float:
    return mean_std(values)[0]


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std of the non-None values (nan where undefined)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return math.nan, math.nan
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, math.nan
    return mu, math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def load_rows(record: dict, root: Path = REPO) -> list[dict]:
    """The per-puzzle rows collected for a record (`collect --rows`)."""
    path = record.get("rows_path")
    if not path:
        raise FileNotFoundError(f"{record['key']}: no rows (collect with --rows)")
    with (root / path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def correct_by_pos(rows: list[dict]) -> list[bool]:
    return [r["status"] == "correct" for r in sorted(rows, key=lambda r: r["pos"])]


def paired_vs(reference: list[dict], arm: list[dict], root: Path = REPO) -> PairedDifference:
    """Arm minus reference, paired by puzzle within each seed and pooled over shared seeds."""
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
    n_text = n_range([r["n"] for r in records if r.get("n") is not None])
    if len(reasons) == 1:
        return "Evaluated on " + EVAL_RULES.get(next(iter(reasons)), "{n} puzzles").format(n=n_text)
    return f"Evaluated sequentially on a fixed random ordering of the test split (n = {n_text})"


def n_range(ns: list[int]) -> str:
    """"1,000" or "800--1,000"; "?" when empty."""
    ns = sorted(set(ns))
    if not ns:
        return "?"
    return f"{ns[0]:,}" if len(ns) == 1 else f"{ns[0]:,}--{ns[-1]:,}"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x: float | None, digits: int = 1) -> str:
    return "--" if _missing(x) else f"{100 * x:.{digits}f}"


def pm(mu: float, sd: float, digits: int = 1) -> str:
    if _missing(mu):
        return "--"
    if _missing(sd):
        return f"{100 * mu:.{digits}f}"
    return f"{100 * mu:.{digits}f} $\\pm$ {100 * sd:.{digits}f}"


def delta_text(d: PairedDifference) -> str:
    if d.n == 0:
        return "--"
    return f"{100 * d.delta:+.1f} [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]"


def num(x, digits: int = 1) -> str:
    if _missing(x):
        return "--"
    return f"{x:,.{digits}f}" if isinstance(x, float) else f"{x:,}"


def escape(text: str) -> str:
    return (str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#"))


@dataclass
class Table:
    """A LaTeX table; a body row of a single cell is a section heading."""

    columns: list[str]                  # header cells (LaTeX)
    rows: list[list[str]]               # body cells (LaTeX)
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


def pdf_bytes(fig) -> bytes:
    """Render a matplotlib figure to PDF bytes and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def baseline_rows(benchmark: str, results_dir: Path = RESULTS) -> list[dict]:
    """Finished baseline runs for `benchmark` as {label, result}; [] before they are collected."""
    try:
        records = read_results("baselines", results_dir)
    except FileNotFoundError:
        return []
    out = []
    for rec in records:
        if rec.get("group") == benchmark and rec.get("status") == "done" and rec.get("result_path"):
            out.append({"label": rec["label"],
                        "result": json.loads((REPO / rec["result_path"]).read_text())})
    return out


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
                    help="also copy outputs here (e.g. path/to/paper/figs)")
    return ap.parse_args()
