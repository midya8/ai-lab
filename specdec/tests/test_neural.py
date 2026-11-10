import numpy as np
from specdec.neural import MiniGPTLM, n_params
from specdec import speculative_generate, autoregressive_generate
from microgrid import MiniGPT


def mk(seed, d, layers):
    return MiniGPTLM(MiniGPT(5, d=d, heads=2, layers=layers, ctx=6, seed=seed), 6)


def test_probs_valid_and_window_only():
    m = mk(0, 8, 1)
    p = m.probs([1, 2, 3])
    assert p.shape == (5,) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
    long = [4, 4, 4, 0, 1, 2, 3, 1, 2]
    assert np.allclose(m.probs(long), m.probs(long[-6:]))


def test_identical_models_accept_everything():
    t = mk(0, 8, 1)
    st = {}
    out = speculative_generate(t, mk(0, 8, 1), [0], 20, 4, np.random.default_rng(0), st)
    assert len(out) == 20 and st["accepted"] == st["proposed"]


def test_exact_distribution_with_neural_pair():
    t, d = mk(0, 8, 1), mk(1, 4, 1)
    N = 3000
    first = np.zeros(5)
    for s in range(N):
        first[speculative_generate(t, d, [0], 1, 3, np.random.default_rng(s))[0]] += 1
    p = t.probs([0])
    exp = p * N
    chi = (((first - exp) ** 2) / exp).sum()
    assert chi < 18.5  # chi2, 4 dof, p~0.001


def test_n_params_ratio():
    assert n_params(mk(0, 16, 2).model) > n_params(mk(0, 8, 1).model)
