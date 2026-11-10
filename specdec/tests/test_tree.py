import numpy as np
from collections import Counter
from specdec import NgramLM, MixtureLM, tree_generate


def models():
    rng = np.random.default_rng(0)
    toks = list(rng.integers(0, 4, 3000))
    t = NgramLM(4, 2, 0.1).fit(toks)
    d = MixtureLM(t, NgramLM(4, 1, 1.0).fit(toks[::-1]), 0.4)
    return t, d


def test_length_and_stats():
    t, d = models()
    st = {}
    out = tree_generate(t, d, [0], 37, 3, 2, np.random.default_rng(1), st)
    assert len(out) == 37 and st["target_calls"] >= 37 / 4 and st["draft_nodes"] > 0


def test_exact_joint_distribution_of_first_two_tokens():
    t, d = models()
    N = 20000
    rng = np.random.default_rng(2)
    c = Counter(tuple(tree_generate(t, d, [1], 2, 3, 2, rng)) for _ in range(N))
    chi2 = 0.0
    for a in range(4):
        pa = t.probs([1])
        for b in range(4):
            e = N * pa[a] * t.probs([1, a])[b]
            chi2 += (c.get((a, b), 0) - e) ** 2 / e
    assert chi2 < 39.3  # df=15, p=0.001


def test_more_candidates_fewer_target_calls():
    t, d = models()
    calls = []
    for m in (1, 4):
        st = {}
        for s in range(20):
            tree_generate(t, d, [0], 60, 4, m, np.random.default_rng(s), st)
        calls.append(st["target_calls"])
    assert calls[1] < calls[0]
