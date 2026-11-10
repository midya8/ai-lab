import numpy as np
from collections import Counter
from specdec import *
from specdec.lookup import PromptLookupLM


def test_probs_valid_and_copies():
    lm = PromptLookupLM(5, n=2, conf=0.8)
    q = lm.probs([1, 2, 3, 0, 1, 2])
    assert abs(q.sum() - 1) < 1e-12 and q.argmax() == 3 and q[3] > 0.8
    assert np.allclose(lm.probs([0, 1]), 0.2)  # too short: uniform
    assert np.allclose(lm.probs([1, 2, 3, 4, 0]), 0.2)  # no match


def test_most_recent_match_wins():
    q = PromptLookupLM(6, n=1, conf=0.9).probs([2, 3, 2, 4, 2])
    assert q.argmax() == 4


def test_exact_with_lookup_draft():
    r = np.random.default_rng(0)
    t = NgramLM(4, 3).fit(list(r.integers(0, 4, 2000)))
    d = PromptLookupLM(4, n=1)
    rng = np.random.default_rng(1)
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


def test_high_acceptance_on_repetitive_target():
    V = 6
    toks = [i % V for i in range(600)]
    t = NgramLM(V, 2, alpha=0.05).fit(toks)
    st = {}
    speculative_generate(t, PromptLookupLM(V, n=2), [0, 1, 2, 3], 60, 4, np.random.default_rng(2), st)
    assert st["accepted"] / st["proposed"] > 0.6
