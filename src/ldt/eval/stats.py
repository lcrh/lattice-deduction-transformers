"""Statistics for sequential evaluation.

All intervals here are for a Bernoulli success rate p estimated from a
prefix of an i.i.d. stream (puzzles drawn in a fixed random order).

- `confidence_sequence`: beta-binomial mixture confidence sequence (Robbins).
  It covers p simultaneously for every n with probability >= 1 - alpha, so it
  stays valid when the number of evaluated puzzles is chosen by a stopping
  rule that looks at the results, and when a stopped run is later extended.
- `binomial_upper`: exact one-sided Clopper-Pearson upper bound, used for the
  wrong-answer rate (a rare-event count reported alongside accuracy).
- `inverse_binomial_rate`: unbiased failure-rate estimate when sampling was
  stopped at the K-th failure (only valid for such runs).
- `paired_difference`: accuracy difference between two runs on the puzzles
  both evaluated (their common prefix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_mixture_ratio(p: float, successes: int, failures: int,
                       prior_a: float, prior_b: float) -> float:
    """log of the beta-binomial mixture martingale evaluated at null p."""
    log_lik = 0.0
    if successes:
        log_lik += successes * math.log(p)
    if failures:
        log_lik += failures * math.log1p(-p)
    return (_log_beta(prior_a + successes, prior_b + failures)
            - _log_beta(prior_a, prior_b) - log_lik)


def _bisect(f, lo: float, hi: float, iters: int = 60) -> float:
    """Point where predicate `f` switches from True (at lo) to False (at hi)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def confidence_sequence(successes: int, n: int, alpha: float = 0.05,
                        prior_a: float = 1.0, prior_b: float = 1.0) -> tuple[float, float]:
    """Anytime-valid (1 - alpha) interval for p after `successes` out of `n`.

    The set {p : M_n(p) < 1/alpha} of the beta-binomial mixture martingale.
    M_n is log-convex in p and minimized at successes / n, so the set is an
    interval found by bisection on each side.
    """
    if n == 0:
        return 0.0, 1.0
    failures = n - successes
    threshold = math.log(1.0 / alpha)
    p_hat = successes / n

    def inside(p: float) -> bool:
        return _log_mixture_ratio(p, successes, failures, prior_a, prior_b) < threshold

    lo = 0.0 if successes == 0 else _bisect(lambda p: not inside(p), 0.0, p_hat)
    hi = 1.0 if failures == 0 else _bisect(inside, p_hat, 1.0)
    return lo, hi


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    m = max(log_terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in log_terms))


def binomial_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1 - alpha) Clopper-Pearson upper bound on p from k/n."""
    if n == 0 or k >= n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    return _bisect(lambda p: _binomial_cdf(k, n, p) > alpha, k / n, 1.0)


def inverse_binomial_rate(failures: int, n: int) -> float:
    """(K - 1) / (n - 1), unbiased when sampling stopped at the K-th failure.

    Only meaningful for runs stopped that way; falls back to failures / n when K < 2.
    """
    if failures >= 2 and n >= 2:
        return (failures - 1) / (n - 1)
    return failures / n if n else 0.0


@dataclass(frozen=True)
class PairedDifference:
    n: int              # common prefix length
    acc_a: float
    acc_b: float
    delta: float        # acc_a - acc_b
    lo: float           # normal-approximation (1 - alpha) interval on delta
    hi: float
    only_a: int         # solved by a, not by b
    only_b: int


def paired_difference(correct_a: list[bool], correct_b: list[bool],
                      alpha: float = 0.05) -> PairedDifference:
    """Paired accuracy difference of two runs over their common prefix of puzzles.

    The interval is a paired (McNemar-style) normal approximation; it is not
    anytime-valid and is meant for ablation tables.
    """
    n = min(len(correct_a), len(correct_b))
    a, b = correct_a[:n], correct_b[:n]
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    if n == 0:
        return PairedDifference(n=0, acc_a=0.0, acc_b=0.0, delta=0.0, lo=-1.0, hi=1.0,
                                only_a=0, only_b=0)
    delta = (only_a - only_b) / n
    var = max((only_a + only_b) - (only_a - only_b) ** 2 / n, 0.0) / n ** 2
    z = _normal_quantile(1.0 - alpha / 2)
    half = z * math.sqrt(var)
    return PairedDifference(
        n=n, acc_a=sum(a) / n, acc_b=sum(b) / n, delta=delta,
        lo=delta - half, hi=delta + half, only_a=only_a, only_b=only_b,
    )


def _normal_quantile(q: float) -> float:
    """Inverse standard normal CDF by bisection on erf."""
    return _bisect(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) < q, -10.0, 10.0)
