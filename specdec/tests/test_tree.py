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


def test_tree_cost_per_token():
    from specdec import tree_cost_per_token
    st = {"target_calls": 10, "draft_nodes": 40}
    assert tree_cost_per_token(st, 40, 0.1) == (10 + 4.0) / 40
    t, d = models()
    st = {}
    tree_generate(t, d, [0], 50, 3, 2, np.random.default_rng(3), st)
    assert tree_cost_per_token(st, 50, 0.0) == st["target_calls"] / 50


def test_wide_trees_lose_when_draft_is_expensive():
    from specdec import tree_cost_per_token
    t, d = models()
    cost = {}
    for m in (1, 8):
        tot = 0.0
        for s in range(10):
            st = {}
            tree_generate(t, d, [0], 60, 4, m, np.random.default_rng(s), st)
            tot += tree_cost_per_token(st, 60, 0.5)
        cost[m] = tot / 10
    assert cost[8] > cost[1]


def test_tempered_and_markov_models():
    import numpy as np
    from specdec import MarkovLM, TemperedLM
    m = MarkovLM([[0.9, 0.1], [0.2, 0.8]])
    assert np.allclose(m.probs([1]), [0.2, 0.8])
    assert np.allclose(TemperedLM(m, 1.0).probs([0]), [0.9, 0.1])
    flat = TemperedLM(m, 4.0).probs([0])
    assert abs(flat.sum() - 1) < 1e-9 and flat[0] < 0.9 and flat[0] > flat[1]
