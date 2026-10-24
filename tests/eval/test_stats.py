import math
import random

from ldt.eval.stats import (
    _binomial_cdf, binomial_upper, confidence_sequence, inverse_binomial_rate,
    paired_difference,
)


def test_binomial_upper_zero_failures_closed_form():
    for n in (10, 1000, 422_786):
        assert math.isclose(binomial_upper(0, n), 1 - 0.05 ** (1 / n), rel_tol=1e-12)


def test_binomial_upper_matches_known_values():
    # One-sided 95% Clopper-Pearson upper bounds (scipy.stats.beta.ppf(0.95, k+1, n-k)).
    assert math.isclose(binomial_upper(1, 10), 0.39416, abs_tol=1e-4)
    assert math.isclose(binomial_upper(5, 10), 0.77756, abs_tol=1e-4)
    for k, n in ((1, 10), (3, 1000), (20, 5000)):
        assert math.isclose(_binomial_cdf(k, n, binomial_upper(k, n)), 0.05, abs_tol=1e-6)


def test_confidence_sequence_contains_estimate_and_shrinks():
    lo1, hi1 = confidence_sequence(950, 1000)
    lo2, hi2 = confidence_sequence(9500, 10000)
    assert lo1 < 0.95 < hi1 and lo2 < 0.95 < hi2
    assert hi2 - lo2 < hi1 - lo1
    assert confidence_sequence(1000, 1000)[1] == 1.0
    assert confidence_sequence(0, 1000)[0] == 0.0


def test_confidence_sequence_anytime_coverage():
    """The interval covers p at every point of a stream with probability >= 1 - alpha.

    So any data-dependent stopping time is covered too.
    """
    rng = random.Random(0)
    n_streams, length, every = 400, 3000, 50
    for p in (0.85, 0.99):
        ever_missed = 0
        for _ in range(n_streams):
            s = 0
            for n in range(1, length + 1):
                s += rng.random() < p
                if n % every == 0:
                    lo, hi = confidence_sequence(s, n, alpha=0.05)
                    if not lo <= p <= hi:
                        ever_missed += 1
                        break
        assert ever_missed / n_streams <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / n_streams)


def test_inverse_binomial_rate_unbiased():
    rng = random.Random(1)
    q, k, trials = 0.1, 5, 20_000
    est = []
    for _ in range(trials):
        n = failures = 0
        while failures < k:
            n += 1
            failures += rng.random() < q
        est.append(inverse_binomial_rate(failures, n))
    mean = sum(est) / trials
    sd = math.sqrt(sum((e - mean) ** 2 for e in est) / (trials - 1))
    assert abs(mean - q) < 4 * sd / math.sqrt(trials)


def test_paired_difference_uses_common_prefix():
    a = [True, True, False, True, True]
    b = [True, False, False, False]
    d = paired_difference(a, b)
    assert d.n == 4 and d.only_a == 2 and d.only_b == 0
    assert math.isclose(d.delta, 0.5)
    assert d.lo < d.delta < d.hi
