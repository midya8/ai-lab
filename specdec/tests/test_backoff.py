import numpy as np
from collections import Counter
from specdec import *


def test_valid_and_longest_match_preferred():
    lm = BackoffLookupLM(6, n_max=3, n_min=1)
    # suffix [1,2,3] seen followed by 4; suffix [3] also seen earlier followed by 5 (most recent for n=1 is 4 anyway)
    ctx = [3, 5, 1, 2, 3, 4, 0, 1, 2, 3]
    q = lm.probs(ctx)
    assert abs(q.sum() - 1) < 1e-12 and q.argmax() == 4 and q[4] > 0.94


def test_backs_off_to_shorter_and_uniform():
    lm = BackoffLookupLM(5, n_max=3, n_min=1)
    q = lm.probs([0, 1, 2, 3, 4, 2])  # only n=1 matches: 2 -> 3
    assert q.argmax() == 3 and abs(q[3] - (lm.conf(1) + (1 - lm.conf(1)) / 5)) < 1e-12
    assert np.allclose(lm.probs([0, 1, 2]), 0.2)
    assert lm.conf(3) > lm.conf(1)


def test_exact_with_backoff_draft():
    r = np.random.default_rng(0)
    t = NgramLM(4, 3).fit(list(r.integers(0, 4, 2000)))
    d = BackoffLookupLM(4, n_max=3)
    rng = np.random.default_rng(2)
    N = 6000
    spec = Counter(tuple(speculative_generate(t, d, [1, 2, 1], 2, 3, rng)) for _ in range(N))
    p1 = t.probs([1, 2, 1])
    chi2 = 0
    for a in range(4):
        p2 = t.probs([1, 2, 1, a])
        for b in range(4):
            e = N * p1[a] * p2[b]
            chi2 += (spec.get((a, b), 0) - e) ** 2 / e
    assert chi2 < 37.6
