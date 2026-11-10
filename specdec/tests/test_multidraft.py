import numpy as np
from collections import Counter
from specdec import *


def pq():
    p = np.array([0.5, 0.3, 0.15, 0.05]); q = np.array([0.1, 0.2, 0.3, 0.4])
    return p, q


def test_m1_acceptance_equals_alpha():
    p, q = pq()
    assert abs(multi_draft_acceptance(p, q, 1) - theoretical_alpha(p, q)) < 1e-12


def test_acceptance_monotone_in_m():
    p, q = pq()
    a = [multi_draft_acceptance(p, q, m) for m in (1, 2, 3, 5, 8)]
    assert all(x < y for x, y in zip(a, a[1:])) and a[-1] <= 1.0


def test_step_distribution_is_exactly_target():
    p, q = pq()
    rng = np.random.default_rng(0)
    N = 40000
    for m in (1, 2, 4):
        c = Counter(multi_draft_step(p, q, m, rng)[0] for _ in range(N))
        chi2 = sum((c.get(i, 0) - N * p[i]) ** 2 / (N * p[i]) for i in range(4))
        assert chi2 < 16.3  # df=3, p=0.001


def test_empirical_acceptance_matches_theory():
    p, q = pq()
    rng = np.random.default_rng(1)
    N = 30000
    got = sum(multi_draft_step(p, q, 3, rng)[1] is not None for _ in range(N)) / N
    assert abs(got - multi_draft_acceptance(p, q, 3)) < 0.01


def test_generate_length_and_stats():
    r = np.random.default_rng(2)
    toks = list(r.integers(0, 5, 2000))
    t = NgramLM(5, 3).fit(toks); d = NgramLM(5, 1, alpha=1.0).fit(toks)
    st = {}
    out = multi_draft_generate(t, d, [0, 1], 50, 3, np.random.default_rng(3), st)
    assert len(out) == 50 and st["steps"] == 50 and 0 < st["accepted"] <= 50
