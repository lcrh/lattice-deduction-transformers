"""E2 tables (S1/S2/S3) + Figure S4 + distribution figures.

    uv run --with matplotlib python followups/search_process/plot_all.py

Reads:
  results/summary.csv                 (from collect.py)
  results/jsonl/<name>.eval.jsonl      (staged per-puzzle rows)

Writes under results/ (tables) and plots/ (figures):
  results/table_s1.md   decision-policy scan: solve rate / batched calls /
                        seq-forwards p50,p90 / resets, per checkpoint.
  results/table_s2.md   matched-vs-mismatched 2x2.
  results/table_s3.md   backtracking policies + unsound_negation_rate column.
  plots/fig_s4.pdf      x=train steps, y=sequential forwards/solve (log),
                        one line per search config.
  plots/forwards_hist.pdf        per-puzzle forwards histograms (baseline vs
                                 best policy).
  plots/conflict_depth_hist.pdf  decision-depth-at-conflict per backtrack policy.

Always reports the README key-cost metrics (solve rate, puzzle_calls, seq-cost,
resets). Degrades gracefully: a config with no data is skipped and the rest
still render. matplotlib uses the Agg backend (no display needed).
"""

from __future__ import annotations

import csv
import json
import os

from followups.search_process import configs as C

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
JSONL_DIR = os.path.join(RESULTS_DIR, "jsonl")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
PLOTS_DIR = os.path.join(HERE, "plots")

written: list[str] = []
skipped: list[str] = []


# --------------------------------------------------------------------------
# Data loading.
# --------------------------------------------------------------------------

def _fnum(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_summary() -> dict[str, dict]:
    """{config: {"seeds": [row,...], "mean": row|None}} from summary.csv."""
    if not os.path.exists(SUMMARY_CSV):
        print(f"[plot] no {SUMMARY_CSV} — run collect.py first.", flush=True)
        return {}
    out: dict[str, dict] = {}
    with open(SUMMARY_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            cfg = row["config"]
            e = out.setdefault(cfg, {"seeds": [], "mean": None})
            if row["seed"] == "mean":
                e["mean"] = row
            elif row["seed"] == "range":
                continue
            else:
                e["seeds"].append(row)
    return out


def load_jsonl(name: str) -> list[dict]:
    path = os.path.join(JSONL_DIR, f"{name}.eval.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _mean(rows: list[dict], col: str):
    vals = [_fnum(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


# --------------------------------------------------------------------------
# Tables.
# --------------------------------------------------------------------------

def table_s1(data: dict) -> None:
    lines = ["# Table S1 — decision-policy scan", "",
             "| checkpoint | cell | digit | solve rate | calls/solve | "
             "puzzle_calls | seq p50 | seq p90 | resets |",
             "|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for base in ("baseline", "base_1k"):
        for tag, cp, dp in C.S1_COMBOS:
            name = f"s1_{base}_{tag}"
            e = data.get(name)
            if not e or not e["seeds"]:
                skipped.append(f"table_s1:{name}")
                continue
            any_row = True
            s = e["seeds"]
            lines.append(
                f"| {base} | {cp} | {dp} | {_fmt(_mean(s,'accuracy'))} | "
                f"{_fmt(_mean(s,'calls_per_solve'),4)} | "
                f"{_fmt(_mean(s,'avg_puzzle_calls'),4)} | "
                f"{_fmt(_mean(s,'seq_forwards_p50'),4)} | "
                f"{_fmt(_mean(s,'seq_forwards_p90'),4)} | "
                f"{_fmt(_mean(s,'avg_resets'))} |")
    _write_table("table_s1.md", lines, any_row)


def table_s2(data: dict) -> None:
    lines = ["# Table S2 — matched vs mismatched (2x2)", "",
             "| train | eval | solve rate | calls/solve | seq p50 | resets |",
             "|---|---|---|---|---|---|"]
    cells = [
        ("P0", "P0", "s2_trainP0_evalP0"),
        ("P0", "P*", "s2_trainP0_evalPstar"),
        ("P*", "P0", "s2_trainPstar_evalP0"),
        ("P*", "P*", "s2_trainPstar_evalPstar"),
    ]
    any_row = False
    for tr, ev, name in cells:
        e = data.get(name)
        if not e or not e["seeds"]:
            skipped.append(f"table_s2:{name}")
            continue
        any_row = True
        s = e["seeds"]
        lines.append(
            f"| train {tr} | eval {ev} | {_fmt(_mean(s,'accuracy'))} | "
            f"{_fmt(_mean(s,'calls_per_solve'),4)} | "
            f"{_fmt(_mean(s,'seq_forwards_p50'),4)} | "
            f"{_fmt(_mean(s,'avg_resets'))} |")
    _write_table("table_s2.md", lines, any_row)


def table_s3(data: dict) -> None:
    lines = ["# Table S3 — backtracking policies", "",
             "| checkpoint | backtrack | solve rate | calls/solve | seq p50 | "
             "resets | mean conflict depth | negations | unsound_negation_rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for base in ("baseline", "base_1k"):
        for tag, flags in C.S3_POLICIES:
            name = f"s3_{base}_{tag}"
            e = data.get(name)
            if not e or not e["seeds"]:
                skipped.append(f"table_s3:{name}")
                continue
            any_row = True
            s = e["seeds"]
            lines.append(
                f"| {base} | {flags['backtrack']} | {_fmt(_mean(s,'accuracy'))} | "
                f"{_fmt(_mean(s,'calls_per_solve'),4)} | "
                f"{_fmt(_mean(s,'seq_forwards_p50'),4)} | "
                f"{_fmt(_mean(s,'avg_resets'))} | "
                f"{_fmt(_mean(s,'conflict_depth_mean'))} | "
                f"{_fmt(_mean(s,'n_negations'),4)} | "
                f"{_fmt(_mean(s,'unsound_negation_rate'))} |")
    _write_table("table_s3.md", lines, any_row)


def _write_table(fname: str, lines: list[str], any_row: bool) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, fname)
    if not any_row:
        lines.append("")
        lines.append("_(no data yet — run collect.py after evals land.)_")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    written.append(path)


# --------------------------------------------------------------------------
# Figures (matplotlib, lazy import so tables still emit without it).
# --------------------------------------------------------------------------

def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc}); skipping figures.",
              flush=True)
        return None


def fig_s4(data: dict, plt) -> None:
    # x = train steps (1K/2K), y = seq forwards/solve (log), line per combo.
    budgets = {"base_1k": 1000, "baseline": 2000}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")
    colors = ["#c0563b", "#2f6f9f", "#4a8b5c", "#8a5fb0"]
    any_series = False
    for i, (tag, flags) in enumerate(C.S4_COMBOS):
        xs, ys = [], []
        for budget, base in C.S4_CKPTS:
            name = f"s4_{budget}_{tag}"
            e = data.get(name)
            if not e or not e["seeds"]:
                continue
            v = _mean(e["seeds"], "seq_forwards_p50")
            if v is None:
                continue
            xs.append(budgets[base]); ys.append(v)
        if len(xs) >= 1:
            order = sorted(range(len(xs)), key=lambda k: xs[k])
            xs = [xs[k] for k in order]; ys = [ys[k] for k in order]
            ax.plot(xs, ys, "-o", color=colors[i % len(colors)], lw=2, label=tag)
            any_series = True
        else:
            skipped.append(f"fig_s4:{tag}")
    if not any_series:
        plt.close(fig)
        skipped.append("fig_s4:ALL(no data)")
        return
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("train steps (log)")
    ax.set_ylabel("sequential forwards / solve, p50 (log)")
    ax.set_title("S4 — search quality x training budget", fontweight="bold")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plt, "fig_s4.pdf")


def fig_forwards_hist(plt) -> None:
    # Per-puzzle forwards histograms: baseline policy vs best policy on base_1k.
    pairs = [
        ("baseline policy", "s1_base_1k_baseline_pol__on__base_1k_seed0"),
        ("mrv+rank_k", "s1_base_1k_mrv_rankk__on__base_1k_seed0"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.patch.set_facecolor("white")
    colors = ["#777777", "#c0563b"]
    any_series = False
    for (label, name), col in zip(pairs, colors):
        rows = load_jsonl(name)
        fwd = [r.get("forwards_seq", r.get("forwards_unbatched"))
               for r in rows if r.get("kind") == "puzzle"]
        fwd = [f for f in fwd if isinstance(f, (int, float)) and f >= 0]
        if not fwd:
            skipped.append(f"forwards_hist:{name}")
            continue
        ax.hist(fwd, bins=30, alpha=0.55, color=col, label=label)
        any_series = True
    if not any_series:
        plt.close(fig)
        skipped.append("forwards_hist:ALL(no data)")
        return
    ax.set_xlabel("per-puzzle sequential forwards")
    ax.set_ylabel("# puzzles")
    ax.set_title("Per-puzzle forwards: baseline vs best policy (base_1k)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, plt, "forwards_hist.pdf")


def fig_conflict_depth_hist(plt) -> None:
    # Decision-depth-at-conflict per backtrack policy (base_1k). Reads the
    # conflict_depths list stored in each eval.json header row of the jsonl,
    # or falls back to the eval.json (not staged) — we use the staged jsonl
    # header row which carries the summary only, so read the full eval.json is
    # not available here; instead histogram from per-puzzle round_solved as a
    # proxy is NOT valid. We therefore read conflict_depths from the summary
    # csv's conflict_depth_mean is a scalar only — so this figure uses the
    # staged jsonl's header 'conflict_depths' if present.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.patch.set_facecolor("white")
    colors = ["#2f6f9f", "#4a8b5c", "#8a5fb0", "#c0563b"]
    any_series = False
    for i, (tag, flags) in enumerate(C.S3_POLICIES):
        if flags["backtrack"] == "root":
            continue  # root never records depths
        name = f"s3_base_1k_{tag}__on__base_1k_seed0"
        rows = load_jsonl(name)
        depths = []
        for r in rows:
            if r.get("kind") == "header":
                depths = r.get("conflict_depths", []) or []
                break
        if not depths:
            skipped.append(f"conflict_depth_hist:{name}")
            continue
        ax.hist(depths, bins=range(0, max(depths) + 2), alpha=0.5,
                color=colors[i % len(colors)], label=flags["backtrack"])
        any_series = True
    if not any_series:
        plt.close(fig)
        skipped.append("conflict_depth_hist:ALL(no data)")
        return
    ax.set_xlabel("decision depth at conflict")
    ax.set_ylabel("# conflicts")
    ax.set_title("Decision-depth-at-conflict per backtrack policy (base_1k)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, plt, "conflict_depth_hist.pdf")


def _save(fig, plt, fname: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    written.append(path)


def main() -> None:
    data = load_summary()
    table_s1(data)
    table_s2(data)
    table_s3(data)
    plt = _import_plt()
    if plt is not None:
        fig_s4(data, plt)
        fig_forwards_hist(plt)
        fig_conflict_depth_hist(plt)

    print("\n=== plot_all summary ===", flush=True)
    if written:
        print("Written:", flush=True)
        for p in written:
            print(f"  {p}", flush=True)
    else:
        print("Nothing written (no data).", flush=True)
    if skipped:
        print("Skipped for lack of data:", flush=True)
        for s in skipped:
            print(f"  {s}", flush=True)


if __name__ == "__main__":
    main()
