"""Portfolio analysis: how many independently trained models it takes to solve every test puzzle.

Inputs (paper/results/portfolio.csv with --rows, from `paper.launch collect`):
anchor models evaluated on the whole test split, and reruns of every model
(independent eval seeds = extra attempts) on the anchors' failure set H.

A puzzle's outcome depends only on its own chains, so for any portfolio that
starts with an anchor, the portfolio's unsolved set is the intersection of
its members' failure sets, all of which lie inside the first anchor's
failure set (a subset of H). Everything below is exact under that rule.

    uv run python -m paper.portfolio extract-failures   # writes H for the rerun entries
    uv run python -m paper.portfolio analyze   # paper/out/portfolio.json, portfolio_residue.pdf
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paper.manifest import REPO, table_hints
from paper.tables.common import OUT, RESULTS, done, load_rows, pdf_bytes, read_results, write_output

FAILURES_FILE = RESULTS / "portfolio_failures.txt"


@dataclass
class Attempt:
    """One eval of one model: its rows by dataset index."""

    model: int                  # training seed
    eval_seed: int
    full: bool                  # covers the whole split (anchor eval)
    rows: dict[int, dict]       # idx -> row

    def failures(self) -> set[int]:
        return {i for i, r in self.rows.items() if r["status"] != "correct"}

    def status(self, idx: int) -> str:
        if idx not in self.rows:
            raise KeyError(f"model {self.model} eval_seed {self.eval_seed} has no row for {idx}; "
                           f"reruns must cover the failure set")
        return self.rows[idx]["status"]


def failure_set(anchors: list[Attempt]) -> list[int]:
    """H: puzzles any anchor failed (wrong or timeout)."""
    return sorted(set().union(*(a.failures() for a in anchors)))


def _attempt_order(a: Attempt) -> tuple[bool, int]:
    """Full (anchor) evals first, then by eval seed."""
    return not a.full, a.eval_seed


def first_attempts(attempts: list[Attempt]) -> dict[int, Attempt]:
    """Each model's first attempt: the full eval for anchors, else eval seed 0."""
    out: dict[int, Attempt] = {}
    for a in sorted(attempts, key=_attempt_order):
        out.setdefault(a.model, a)
    return out


def residue_curve(order: list[Attempt]) -> list[int]:
    """Unsolved puzzles after each prefix of `order` (order[0] must be full)."""
    if not order[0].full:
        raise ValueError("a portfolio must start with an anchor (full-split) attempt")
    unsolved = order[0].failures()
    sizes = [len(unsolved)]
    for a in order[1:]:
        unsolved = {i for i in unsolved if a.status(i) != "correct"}
        sizes.append(len(unsolved))
    return sizes


def random_orders(firsts: dict[int, Attempt], n_orders: int, rng: random.Random
                  ) -> list[list[Attempt]]:
    anchors = [a for a in firsts.values() if a.full]
    out = []
    for _ in range(n_orders):
        head = rng.choice(anchors)
        rest = [a for m, a in firsts.items() if m != head.model]
        rng.shuffle(rest)
        out.append([head, *rest])
    return out


def mean_curve(curves: list[list[int]]) -> list[float]:
    return [sum(c[k] for c in curves) / len(curves) for k in range(len(curves[0]))]


def bootstrap_curve(firsts: dict[int, Attempt], n_boot: int, n_orders: int,
                    rng: random.Random) -> tuple[list[float], list[float]]:
    """95% bootstrap band of the mean residue curve (models resampled; first member an anchor)."""
    models = list(firsts.values())
    anchors = [a for a in models if a.full]
    n = len(models)
    boots = []
    for _ in range(n_boot):
        curves = []
        for _ in range(n_orders):
            order = [rng.choice(anchors)] + [rng.choice(models) for _ in range(n - 1)]
            curves.append(residue_curve(order))
        boots.append(mean_curve(curves))
    lo, hi = [], []
    for k in range(n):
        vals = sorted(b[k] for b in boots)
        lo.append(vals[int(0.025 * (len(vals) - 1))])
        hi.append(vals[int(0.975 * (len(vals) - 1))])
    return lo, hi


def attempts_curve(attempts: list[Attempt]) -> list[float]:
    """Residue after k eval seeds of one anchor model, averaged over anchors (vs. k models)."""
    curves = []
    for model in sorted({a.model for a in attempts if a.full}):
        own = sorted((a for a in attempts if a.model == model), key=_attempt_order)
        curves.append(residue_curve(own))
    k = min(len(c) for c in curves)
    return mean_curve([c[:k] for c in curves])


def select(order: list[Attempt], h: list[int], rule: Literal["first", "majority"]) -> Counter:
    """Outcomes over H of a selection rule that needs no ground truth.

    `first` takes the first non-timeout answer in portfolio order; `majority`
    the most frequent one, abstaining on ties. Puzzles outside H count as
    correct (the anchor solved them; other members were not run there).
    """
    counts: Counter = Counter()
    for i in h:
        answers = [(a.rows[i]["solution"], a.rows[i]["status"]) for a in order
                   if a.status(i) != "timeout"]
        if not answers:
            counts["unsolved"] += 1
            continue
        if rule == "first":
            counts[answers[0][1]] += 1
            continue
        votes = Counter(sol for sol, _ in answers).most_common()
        if len(votes) > 1 and votes[0][1] == votes[1][1]:
            counts["unsolved"] += 1
        else:
            counts[next(st for sol, st in answers if sol == votes[0][0])] += 1
    return counts


def solve_probability(attempts: list[Attempt], h: list[int]) -> dict[int, tuple[int, int]]:
    """idx -> (attempts that solved it, attempts that ran it)."""
    out = {}
    for i in h:
        ran = [a for a in attempts if i in a.rows]
        out[i] = (sum(a.rows[i]["status"] == "correct" for a in ran), len(ran))
    return out


def analyze(attempts: list[Attempt], n_total: int, n_orders: int = 200, n_boot: int = 200,
            seed: int = 0) -> dict:
    rng = random.Random(seed)
    anchors = [a for a in attempts if a.full]
    h = failure_set(anchors)
    firsts = first_attempts(attempts)
    orders = random_orders(firsts, n_orders, rng)
    curves = [residue_curve(o) for o in orders]
    natural = sorted(firsts.values(), key=lambda a: (not a.full, a.model))
    natural_curve = residue_curve(natural)
    zero_frac = [sum(c[k] == 0 for c in curves) / len(curves) for k in range(len(natural))]
    lo, hi = bootstrap_curve(firsts, n_boot, max(1, n_orders // 10), rng)
    probs = solve_probability(attempts, h)
    hardest = sorted(probs.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))[:20]
    return {
        "n_total": n_total,
        "n_models": len(firsts),
        "n_anchors": len(anchors),
        "failure_set_size": len(h),
        "residue_mean": mean_curve(curves),
        "residue_ci95": [lo, hi],
        "residue_natural_order": natural_curve,
        "fraction_of_orders_at_zero": zero_frac,
        "smallest_k": next((k + 1 for k, r in enumerate(natural_curve) if r == 0), None),
        "attempts_curve": attempts_curve(attempts),
        "selection": {rule: [dict(select(natural[:k], h, rule)) for k in range(1, len(natural) + 1)]
                      for rule in ("first", "majority")},
        "hardest": [{"idx": i, "solved": s, "attempts": n} for i, (s, n) in hardest],
    }


def plot(result: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt     # noqa: E402

    k = list(range(1, len(result["residue_mean"]) + 1))
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(k, result["residue_mean"], marker="o", ms=3, color="0.2", label="k models x 1 attempt")
    ax.fill_between(k, *result["residue_ci95"], alpha=0.2, color="0.2")
    att = result["attempts_curve"]
    ax.plot(range(1, len(att) + 1), att, marker="s", ms=3, ls="--", label="1 model x k attempts")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Portfolio size k")
    ax.set_ylabel("Unsolved test puzzles")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return pdf_bytes(fig)


def load_attempts(records: list[dict], anchors_entry: str, rerun_entries: list[str],
                  root: Path = REPO) -> list[Attempt]:
    out = []
    for rec in done(records):
        if rec["entry"] not in (anchors_entry, *rerun_entries):
            continue
        rows = {r["idx"]: r for r in load_rows(rec, root)}
        out.append(Attempt(model=rec["seed"], eval_seed=rec["eval_seed"] or 0,
                           full=rec["entry"] == anchors_entry, rows=rows))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paper.portfolio")
    ap.add_argument("cmd", choices=("extract-failures", "analyze"))
    ap.add_argument("--manifest", default="portfolio")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=FAILURES_FILE)
    ap.add_argument("--copy-to", type=Path, default=None)
    args = ap.parse_args(argv)

    hints = table_hints(args.manifest)
    records = read_results(args.manifest, args.results)
    attempts = load_attempts(records, hints["anchors"], hints["reruns"])
    anchors = [a for a in attempts if a.full]
    if not anchors:
        raise SystemExit("no finished anchor evals with rows; run `paper.launch collect "
                         f"{args.manifest} --rows` first")
    if args.cmd == "extract-failures":
        h = failure_set(anchors)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(f"{i}\n" for i in h))
        print(f"wrote {len(h)} indices from {len(anchors)} anchors to {args.out}")
        return
    result = analyze(attempts, n_total=max(len(a.rows) for a in anchors))
    write_output("portfolio.json", json.dumps(result, indent=2), OUT, None)
    path = write_output("portfolio_residue.pdf", plot(result), OUT, args.copy_to)
    print(f"failure set {result['failure_set_size']}, residue by k (natural order): "
          f"{result['residue_natural_order']}\nwrote {path}")


if __name__ == "__main__":
    main()
