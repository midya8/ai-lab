import numpy as np
from collections import Counter
from specdec import *


def toy(seed=0, V=5):
    r = np.random.default_rng(seed)
    toks = list(r.integers(0, V, 3000))
    t = NgramLM(V, 3).fit(toks)
    d = NgramLM(V, 1, alpha=1.0).fit(toks)
    return t, d


def test_identical_models_accept_everything():
    t, _ = toy()
    st = {}
    out = speculative_generate(t, t, [0, 1], 40, 4, np.random.default_rng(1), st)
    assert len(out) == 40 and st["accepted"] == st["proposed"]


def test_exact_length_for_any_k():
    t, d = toy()
    for k in (1, 2, 3, 7):
        assert len(speculative_generate(t, d, [0], 11, k, np.random.default_rng(k))) == 11


def test_output_distribution_matches_target():
    # joint distribution over first 2 tokens must equal autoregressive's
    t, d = toy(3, 4)
    rng = np.random.default_rng(5)
    N = 6000
    spec = Counter(tuple(speculative_generate(t, d, [1, 2], 2, 3, rng)) for _ in range(N))
    exact = {}
    p1 = t.probs([1, 2])
    for a in range(4):
        p2 = t.probs([1, 2, a])
        for b in range(4):
            exact[(a, b)] = p1[a] * p2[b]
    chi2 = sum((spec.get(k, 0) - N * v) ** 2 / (N * v) for k, v in exact.items())
    assert chi2 < 37.6  # chi-square df=15, p=0.001


def test_fewer_target_calls_than_tokens_with_good_draft():
    t, d = toy()
    good = MixtureLM(t, d, 0.9)
    st = {}
    speculative_generate(t, good, [0], 200, 4, np.random.default_rng(2), st)
    assert st["target_calls"] < 200 / 2


def test_alpha_formula_matches_empirical_one_step():
    t, d = toy(7)
    p, q = t.probs([0, 1]), d.probs([0, 1])
    a = theoretical_alpha(p, q)
    rng = np.random.default_rng(9)
    acc = 0
    N = 20000
    for _ in range(N):
        x = int(rng.choice(len(q), p=q))
        acc += rng.random() < min(1, p[x] / q[x])
    assert abs(acc / N - a) < 0.02


def test_expected_speedup_bounds():
    assert expected_speedup(1.0, 4, 0.0) == 5
    assert expected_speedup(0.0, 4, 0.1) < 1.0
    assert expected_speedup(0.8, 4, 0.05) > 2


def test_adaptive_k_shrinks_for_weak_draft_and_grows_for_strong():
    t, d = toy()
    weak, strong = AdaptiveK(0.1), AdaptiveK(0.1)
    for _ in range(30):
        weak.update(0, weak.k(), True)      # always rejected at first token
        strong.update(strong.k(), strong.k(), False)
    assert weak.k() == 1 and strong.k() == 16


def test_adaptive_k_exact_length_and_distribution():
    t, d = toy(3, 4)
    rng = np.random.default_rng(9)
    assert len(speculative_generate(t, d, [0], 25, 0, rng, controller=AdaptiveK())) == 25
    N = 6000
    spec = Counter(tuple(speculative_generate(t, d, [1, 2], 2, 0, rng, controller=AdaptiveK())) for _ in range(N))
    p1 = t.probs([1, 2])
    chi2 = 0.0
    for a in range(4):
        p2 = t.probs([1, 2, a])
        for b in range(4):
            e = N * p1[a] * p2[b]
            chi2 += (spec.get((a, b), 0) - e) ** 2 / e
    assert chi2 < 37.6
