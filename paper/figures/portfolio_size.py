"""Portfolio size: measured and derived failure rate of majority voting against
the number of models, on the test split (manifest `portfolio`, entry
`same_data_test`, with every 1,000-round timeout rerun at 10,000 rounds).

Measured: majority voting over every m-model subset of the 15 models (or 300
random ones), wrong answers plus unresolved puzzles per puzzle.

Derived: each puzzle i has a probability p_i that a model does not answer it
correctly within 10,000 rounds; a failure is a wrong answer with probability
r (the pooled share of failures that are wrong) and otherwise an abstention.
p is given a Beta(a, b) prior, fit by maximum likelihood to how many of the
15 models fail each puzzle (a beta-binomial), and each puzzle's posterior is
Beta(a + f_i, b + n - f_i). Drawing m new models from it, majority voting
fails when no model answers correctly, or when exactly one does and another
returns a (distinct) wrong grid, a tie. Its probability has a closed form in
E[p^m] = B(a + f_i + m, b + n - f_i) / B(a + f_i, b + n - f_i), which equals
the limit of simulating many draws; the derived rate is its mean over puzzles.

Uncertainty: a bootstrap over puzzles and over models (resampled with
replacement, prior refit each time) gives a 90% band. Out-of-sample check:
the same model fit on the 100,000 held-out puzzles (1,000 rounds) is compared
with majority voting measured on the test split at 1,000 rounds.

    uv run python -m paper.figures.portfolio_size [--copy-to DIR]
"""

from __future__ import annotations

import itertools
from collections import Counter
from math import lgamma

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from paper.portfolio import Evals, _resolve, load_evals, test_evals, test_failures  # noqa: E402
from paper.tables.common import cli, pdf_bytes, read_results, write_output  # noqa: E402

MAX_SUBSETS = 300
BOOTSTRAP = 200
M_MAX = 200
LEVELS = {1e-6: "one per million", 1e-9: "one per billion"}


def betaln(x: float, y: float) -> float:
    return lgamma(x) + lgamma(y) - lgamma(x + y)


def measured(ev: Evals, total: int, seed: int = 0) -> dict[int, float]:
    """{m: failures per puzzle} of majority voting, averaged over m-model subsets."""
    rng = np.random.default_rng(seed)
    pool = range(ev.answer.shape[0])
    out = {}
    for m in pool:
        m += 1
        subsets = [list(c) for c in itertools.combinations(pool, m)]
        if len(subsets) > MAX_SUBSETS:
            subsets = [list(rng.choice(len(pool), m, replace=False)) for _ in range(MAX_SUBSETS)]
        rates = []
        for s in subsets:
            chosen = _resolve(ev.answer[s], ev.calls[s], "majority")[0] if m > 1 else ev.answer[s[0]]
            rates.append((chosen != 0).sum() / total)
        out[m] = float(np.mean(rates))
    return out


def fit_prior(counts: Counter, n: int) -> tuple[float, float]:
    """Maximum-likelihood Beta(a, b) for the beta-binomial counts {k: puzzles}."""
    def nll(la: float, lb: float) -> float:
        a, b = np.exp(la), np.exp(lb)
        return -sum(c * (betaln(k + a, n - k + b) - betaln(a, b)) for k, c in counts.items())

    best = min(((nll(la, lb), la, lb) for la in np.linspace(-12, 2, 57) for lb in np.linspace(-4, 8, 49)))
    _, la, lb = best
    step = 0.25
    while step > 1e-6:      # coordinate refinement around the grid optimum
        moves = [(la + dx, lb + dy) for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]
        cand = min(moves, key=lambda p: nll(*p))
        if nll(*cand) < nll(la, lb):
            la, lb = cand
        else:
            step /= 2
    return float(np.exp(la)), float(np.exp(lb))


def derived(counts: Counter, n: int, a: float, b: float, r: float, m: int) -> float:
    """Mean over puzzles of the posterior probability that majority voting over m new models fails."""
    def ep(f: int, j: int) -> float:
        return float(np.exp(betaln(a + f + j, b + n - f) - betaln(a + f, b + n - f)))

    total = sum(counts.values())
    fail = 0.0
    for f, c in counts.items():
        none_correct = ep(f, m)
        tie = m * (ep(f, m - 1) - ep(f, m)) * (1 - (1 - r) ** (m - 1)) if m > 1 else 0.0
        fail += c * (none_correct + tie)
    return fail / total


def derived_curve(answer: np.ndarray, total: int) -> tuple[dict[int, float], tuple[float, float, float]]:
    """Derived majority failure rate for m = 1..M_MAX from an answer matrix
    [model, puzzle] over the puzzles some model fails; the other `total - columns`
    puzzles are solved by every model. Also returns the prior (a, b) and r."""
    n = answer.shape[0]
    fails = (answer != 0).sum(axis=0)
    counts = Counter(fails.tolist())
    counts[0] += total - answer.shape[1]
    r = float((answer > 0).sum() / max(fails.sum(), 1))
    a, b = fit_prior(counts, n)
    return {m: derived(counts, n, a, b, r, m) for m in range(1, M_MAX + 1)}, (a, b, r)


def bootstrap_band(answer: np.ndarray, total: int, seed: int = 0) -> tuple[dict, dict]:
    """5th and 95th percentile of the derived curve, resampling puzzles and models."""
    rng = np.random.default_rng(seed)
    n, hard = answer.shape
    curves = []
    for _ in range(BOOTSTRAP):
        models = rng.integers(0, n, n)
        cols = rng.integers(0, hard, rng.binomial(total, hard / total))
        curves.append(derived_curve(answer[np.ix_(models, cols)], total)[0])
    lo = {m: float(np.percentile([c[m] for c in curves], 5)) for m in curves[0]}
    hi = {m: float(np.percentile([c[m] for c in curves], 95)) for m in curves[0]}
    return lo, hi


def heldout_check(manifest: str, results) -> None:
    """Derived rates fit on the held-out set against majority voting measured on the
    test split, both at 1,000 rounds."""
    held = load_evals(read_results(manifest, results))
    pool = held.pool("same_data/")
    curve = derived_curve(held.answer[pool], held.answer.shape[1])[0]
    fails, total = test_failures(read_results(manifest, results))
    hard = sorted(set().union(*fails.values()))
    answer = np.array([[fails[s].get(i, 0) for i in hard] for s in sorted(fails)])
    zeros = np.zeros_like(answer)
    meas = measured(Evals([str(s) for s in sorted(fails)], answer, zeros, zeros), total)
    print("out-of-sample check at 1,000 rounds (fit: held-out; measured: test):")
    for m in (1, 2, 3, 5, 8, 10, 13):
        print(f"  m={m:2d}  predicted {curve[m]:.2e}  measured {meas[m]:.2e}  ratio {meas[m] / curve[m]:.2f}")


def plot(meas: dict[int, float], curve: dict[int, float], band: tuple[dict, dict],
         pool_size: int) -> bytes:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ms = list(curve)
    ax.fill_between(ms, [band[0][m] for m in ms], [band[1][m] for m in ms], color="0.8", lw=0, zorder=1,
                    label="derived, 90% bootstrap band")
    ax.plot(ms, [curve[m] for m in ms], "-", color="0.35", lw=1.2, label="derived")
    pts = [m for m in meas if meas[m] > 0]
    ax.plot(pts, [meas[m] for m in pts], "o", ms=3.5, color="C3", label="measured (majority, test)")
    for level, text in LEVELS.items():
        ax.axhline(level, color="0.75", lw=0.8, ls=":")
        ax.text(M_MAX, level * 1.3, text, ha="right", va="bottom", fontsize=6.5, color="0.45")
    ax.axvspan(pool_size + 0.5, M_MAX, color="0.95", lw=0, zorder=0)     # beyond the trained models
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200], ["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlim(0.9, M_MAX)
    ax.set_ylim(1e-12, 5e-4)
    ax.set_xlabel("Models in the portfolio")
    ax.set_ylabel("Failures per puzzle")
    ax.legend(fontsize=6.5, loc="lower left", framealpha=1, edgecolor="none")
    fig.tight_layout()
    return pdf_bytes(fig)


def main() -> None:
    args = cli("Portfolio size figure.", default_manifest="portfolio")
    ev, total = test_evals(read_results(args.manifest, args.results))
    n = ev.answer.shape[0]
    curve, (a, b, r) = derived_curve(ev.answer, total)
    band = bootstrap_band(ev.answer, total)
    meas = measured(ev, total)
    print(f"prior Beta({a:.4g}, {b:.4g}); wrong share of failures {r:.3f}")
    for m in range(1, n + 1):
        print(f"m={m:2d}  measured {meas[m]:.2e}  derived {curve[m]:.2e} "
              f"[{band[0][m]:.2e}, {band[1][m]:.2e}]")
    for level, text in LEVELS.items():
        cross = [next((m for m in c if c[m] < level), None) for c in (band[0], curve, band[1])]
        print(f"  derived rate below {text}: m = {cross[1]} (band {cross[0]}-{cross[2]})")
    heldout_check(args.manifest, args.results)
    pdf = plot(meas, curve, band, n)
    print(f"wrote {write_output('portfolio_size.pdf', pdf, args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
