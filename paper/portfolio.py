"""Portfolio strategies, simulated from per-model evals on the held-out set.

Every portfolio model is evaluated once on `sudoku_holdout` at the full
round budget (manifest `portfolio`). A row records whether the model solved
the puzzle, in which round, its answer and its model calls, so any shorter
budget R can be read off the same eval: a model "answers within R" if it did
not time out and solved by round R. Cost under a budget R is approximated as
min(calls, R + 1) model calls (one call per round plus the initial deduction).

A strategy runs a random order of m distinct models from a pool:
  1. Prefilter: the first p models at budget R. If all p answer and agree,
     that answer is accepted. p = 0 skips the prefilter.
  2. Otherwise ("don't know"), the remaining m - p models resolve the puzzle
     at the full budget with a rule: `first` (first answer in order),
     `majority` (most frequent answer among all remaining models; ties stay
     unresolved), or `agree<k>` (run models in order until one answer has k
     votes).
Bootstrap: B random orders per (pool, m), the same orders for every strategy.

    uv run python -m paper.launch collect portfolio --rows
    uv run python -m paper.portfolio simulate [--draws 100] [--out paper/out/portfolio_sim.csv]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lattice_diffusion.eval.stats import binomial_upper

from paper.manifest import REPO
from paper.tables.common import OUT, RESULTS, done, load_rows, read_results

TIMEOUT = -1        # answer id of a timeout; 0 = the correct answer; > 0 = a wrong answer
ROUND_BUDGETS = (1, 2, 5, 10, 20, 50, 100, 1000)
RULES = ("first", "majority", "agree2", "agree3")


@dataclass
class Evals:
    """Per-model outcomes on a common set of puzzles, as [model, puzzle] arrays."""

    names: list[str]          # one per model, e.g. "same_data/seed3"
    answer: np.ndarray        # answer id (see TIMEOUT)
    rounds: np.ndarray        # round solved in (timeouts: the budget)
    calls: np.ndarray         # model calls spent on the puzzle

    def pool(self, prefix: str) -> list[int]:
        return [i for i, n in enumerate(self.names) if n.startswith(prefix)]


def answer_id(row: dict) -> int:
    """0 for a correct answer, a positive id per distinct wrong answer, TIMEOUT otherwise."""
    if row["status"] == "correct":
        return 0
    if row["status"] == "timeout":
        return TIMEOUT
    return 1 + zlib.crc32(row["solution"].encode()) % (2**30)


def build_evals(models: dict[str, list[dict]]) -> Evals:
    """Stack each model's rows on the puzzles every model evaluated."""
    common = sorted(set.intersection(*(set(r["idx"] for r in rows) for rows in models.values())))
    names = sorted(models)
    answer = np.empty((len(names), len(common)), dtype=np.int64)
    rounds = np.empty_like(answer)
    calls = np.empty_like(answer)
    for m, name in enumerate(names):
        by_idx = {r["idx"]: r for r in models[name]}
        for j, idx in enumerate(common):
            r = by_idx[idx]
            answer[m, j], rounds[m, j], calls[m, j] = answer_id(r), r["rounds"], r["calls"]
    return Evals(names, answer, rounds, calls)


@dataclass(frozen=True)
class Strategy:
    m: int              # models in the portfolio
    p: int              # prefilter models (0 = no prefilter)
    budget: int         # prefilter round budget R
    rule: str           # resolution rule for don't-know puzzles


@dataclass
class Outcome:
    correct: int
    wrong: int
    unresolved: int
    calls: float        # total model calls


def _resolve_one(ans: np.ndarray, calls: np.ndarray, rule: str) -> tuple[int, float]:
    """`_resolve` for one puzzle column, exactly (used where wrong answers can collide)."""
    if rule == "majority":
        votes = ans[ans != TIMEOUT]
        vals, counts = np.unique(votes, return_counts=True)
        if votes.size and (counts == counts.max()).sum() == 1:
            return int(vals[counts.argmax()]), float(calls.sum())
        return TIMEOUT, float(calls.sum())
    k = int(rule.removeprefix("agree"))
    seen: dict[int, int] = {}
    for i, a in enumerate(ans.tolist()):
        if a != TIMEOUT:
            seen[a] = seen.get(a, 0) + 1
            if seen[a] == k:
                return a, float(calls[: i + 1].sum())
    return TIMEOUT, float(calls.sum())


def _resolve(ans: np.ndarray, calls: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """Chosen answer (TIMEOUT = unresolved) and calls spent, per puzzle column.

    Vectorized for the common case of at most one wrong vote per puzzle;
    puzzles with two or more wrong votes are resolved exactly one by one.
    """
    n_models, n = ans.shape
    cols = np.arange(n)
    cum_calls = np.cumsum(calls, axis=0)
    if rule == "first":
        answered = ans != TIMEOUT
        has = answered.any(axis=0)
        pos = np.where(has, answered.argmax(axis=0), n_models - 1)
        return np.where(has, ans[pos, cols], TIMEOUT), cum_calls[pos, cols].astype(float)
    right = (ans == 0).sum(axis=0)
    wrong_votes = ans > 0
    n_wrong = wrong_votes.sum(axis=0)
    some_wrong = ans[wrong_votes.argmax(axis=0), cols]      # the wrong answer, if exactly one
    spent = cum_calls[-1].astype(float)
    if rule == "majority":
        chosen = np.where(right > n_wrong, 0, np.where(n_wrong > right, some_wrong, TIMEOUT))
    else:
        k = int(rule.removeprefix("agree"))
        reached = right >= k
        pos = np.argmax(np.cumsum(ans == 0, axis=0) >= k, axis=0)
        chosen = np.where(reached, 0, TIMEOUT)
        spent = np.where(reached, cum_calls[pos, cols], spent).astype(float)
    for j in np.flatnonzero(n_wrong >= 2):
        chosen[j], spent[j] = _resolve_one(ans[:, j], calls[:, j], rule)
    return chosen, spent


def run_strategy(ev: Evals, order: list[int], s: Strategy) -> Outcome:
    """Outcome of strategy `s` with models in `order` (len(order) == s.m)."""
    n = ev.answer.shape[1]
    final = np.full(n, TIMEOUT)
    total_calls = 0.0
    unknown = np.ones(n, dtype=bool)
    if s.p:
        pre = order[: s.p]
        ans = np.where(ev.rounds[pre] <= s.budget, ev.answer[pre], TIMEOUT)
        total_calls += np.minimum(ev.calls[pre], s.budget + 1).sum()
        agree = (ans[0] != TIMEOUT) & (ans == ans[0]).all(axis=0)
        final[agree] = ans[0, agree]
        unknown = ~agree
    rest = order[s.p:]
    if rest and unknown.any():
        cols = np.flatnonzero(unknown)
        chosen, spent = _resolve(ev.answer[np.ix_(rest, cols)], ev.calls[np.ix_(rest, cols)],
                                 s.rule)
        final[cols] = chosen
        total_calls += spent.sum()
    return Outcome(correct=int((final == 0).sum()), wrong=int((final > 0).sum()),
                   unresolved=int((final == TIMEOUT).sum()), calls=float(total_calls))


def strategies(max_m: int) -> list[Strategy]:
    out = []
    for m, p in itertools.product(range(1, max_m + 1), range(0, 4)):
        if p > m:
            continue
        budgets = ROUND_BUDGETS if p else (1000,)
        rules = RULES if p < m else ("first",)
        out += [Strategy(m, p, b, r) for b in budgets for r in rules]
    return out


def simulate(ev: Evals, pools: dict[str, list[int]], draws: int, max_m: int,
             seed: int = 0) -> list[dict]:
    """Bootstrap every strategy on every pool; one summary row per (pool, strategy)."""
    rng = np.random.default_rng(seed)
    n = ev.answer.shape[1]
    out = []
    for pool_name, pool in pools.items():
        orders = [list(rng.permutation(pool)) for _ in range(draws)]
        for s in strategies(min(max_m, len(pool))):
            res = [run_strategy(ev, o[: s.m], s) for o in orders]
            wrong = np.array([r.wrong for r in res])
            answered = sum(r.correct + r.wrong for r in res)
            out.append({
                "pool": pool_name, **asdict(s), "n": n, "draws": draws,
                "coverage": float(np.mean([(r.correct + r.wrong) / n for r in res])),
                "wrong_mean": float(wrong.mean()), "wrong_max": int(wrong.max()),
                # Pooled over draws: draws reuse models, so this is optimistic.
                "wrong_rate_upper": binomial_upper(int(wrong.sum()), answered) if answered else 1.0,
                "unresolved_mean": float(np.mean([r.unresolved for r in res])),
                "calls_per_puzzle": float(np.mean([r.calls / n for r in res])),
            })
    return out


def load_evals(records: list[dict], root: Path = REPO) -> Evals:
    models = {f"{r['entry']}/seed{r['seed']}": load_rows(r, root) for r in done(records)}
    if not models:
        raise SystemExit("no finished portfolio evals with rows; run "
                         "`paper.launch collect portfolio --rows` first")
    return build_evals(models)


def summary_text(rows: list[dict], top: int = 12) -> str:
    """Per pool: strategies with no wrong answer in any draw, by unresolved then calls."""
    lines = []
    for pool in sorted({r["pool"] for r in rows}):
        safe = sorted((r for r in rows if r["pool"] == pool and r["wrong_max"] == 0),
                      key=lambda r: (r["unresolved_mean"], r["calls_per_puzzle"]))
        lines.append(f"== {pool}: {len(safe)} strategies with 0 wrong in every draw")
        lines.append(f"{'m':>3} {'p':>2} {'R':>5} {'rule':>8} {'coverage':>9} "
                     f"{'unresolved':>10} {'calls/puzzle':>12}")
        for r in safe[:top]:
            lines.append(f"{r['m']:>3} {r['p']:>2} {r['budget']:>5} {r['rule']:>8} "
                         f"{r['coverage']:>9.6f} {r['unresolved_mean']:>10.2f} "
                         f"{r['calls_per_puzzle']:>12.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paper.portfolio")
    ap.add_argument("cmd", choices=("simulate",))
    ap.add_argument("--manifest", default="portfolio")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--max-models", type=int, default=13)
    ap.add_argument("--out", type=Path, default=OUT / "portfolio_sim.csv")
    args = ap.parse_args(argv)

    ev = load_evals(read_results(args.manifest, args.results))
    # diff_data: one model per training set, including data seed 42 (same_data seed 0).
    pools = {"same_data": ev.pool("same_data/"),
             "diff_data": ev.pool("diff_data_") + ev.pool("same_data/seed0"),
             "mixed": list(range(len(ev.names)))}
    pools = {k: v for k, v in pools.items() if v}
    rows = simulate(ev, pools, args.draws, args.max_models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(summary_text(rows))
    print(f"wrote {args.out} ({len(rows)} strategies, {ev.answer.shape[1]:,} puzzles, "
          f"{len(ev.names)} models)")


if __name__ == "__main__":
    main()
