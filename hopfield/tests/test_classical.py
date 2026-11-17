import numpy as np
import pytest

from hopfield import classical as C
from hopfield.patterns import random_patterns, correlated_patterns, flip_noise


def rng(s=0):
    return np.random.default_rng(s)


def test_hebbian_symmetric_zero_diag():
    X = random_patterns(5, 40, rng())
    W = C.hebbian(X)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)


def test_rejects_non_pm1():
    with pytest.raises(ValueError):
        C.hebbian(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError):
        C.hebbian(np.ones(4))


def test_pseudoinverse_is_projector():
    X = random_patterns(8, 50, rng(1))
    W = C.pseudoinverse(X, zero_diag=False)
    assert np.allclose(W @ W, W, atol=1e-8)
    assert np.allclose(X @ W, X, atol=1e-8)       # every stored pattern is an exact eigenvector


def test_storkey_matches_naive_loop():
    X = random_patterns(6, 12, rng(2))
    n = 12
    W = np.zeros((n, n))
    for x in X:
        Wn = W.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                h_ij = sum(W[i, k] * x[k] for k in range(n) if k not in (i, j))
                h_ji = sum(W[j, k] * x[k] for k in range(n) if k not in (i, j))
                Wn[i, j] = W[i, j] + (x[i] * x[j] - x[i] * h_ji - h_ij * x[j]) / n
        W = Wn
    assert np.allclose(C.storkey(X), W, atol=1e-12)


def test_storkey_symmetric_and_stabilises_patterns():
    X = random_patterns(15, 100, rng(3))
    W = C.storkey(X)
    assert np.allclose(W, W.T)
    assert C.is_fixed_point(W, X).all()


def test_hebbian_stable_at_low_load_fails_at_high_load():
    low = random_patterns(5, 200, rng(4))
    assert C.is_fixed_point(C.hebbian(low), low).all()
    high = random_patterns(60, 200, rng(5))           # alpha = 0.3 > 0.138
    assert C.is_fixed_point(C.hebbian(high), high).mean() < 0.2


def test_pseudoinverse_beats_hebbian_on_correlated_patterns():
    X = correlated_patterns(10, 100, 0.4, rng(6))
    assert C.is_fixed_point(C.pseudoinverse(X), X).all()
    assert C.is_fixed_point(C.hebbian(X), X).mean() < 0.5


def test_energy_matches_formula_and_ignores_diagonal():
    r = rng(7)
    W = r.normal(size=(9, 9)); W = W + W.T
    s = r.choice([-1.0, 1.0], size=(3, 9))
    Wz = W.copy(); np.fill_diagonal(Wz, 0)
    manual = np.array([-0.5 * x @ Wz @ x for x in s])
    assert np.allclose(C.energy(W, s), manual)


def test_energy_never_increases_per_single_neuron_update():
    r = rng(8)
    X = random_patterns(12, 40, r)
    W = C.hebbian(X)
    s = r.choice([-1.0, 1.0], size=40)
    e = C.energy(W, s)[0]
    for i in r.permutation(40).tolist() * 3:
        h = W[i] @ s
        if h != 0:
            s[i] = np.sign(h)
        e_new = C.energy(W, s)[0]
        assert e_new <= e + 1e-12
        e = e_new


def test_retrieval_energy_trace_monotone_and_converges():
    r = rng(9)
    X = random_patterns(10, 80, r)
    W = C.hebbian(X)
    res = C.retrieve(W, r.choice([-1.0, 1.0], size=(6, 80)), rng=r)
    assert res.converged.all()
    assert np.all(np.diff(res.energies, axis=0) <= 1e-12)
    # a converged state is a fixed point of the dynamics
    again = C.retrieve(W, res.states, max_sweeps=2, rng=r)
    assert np.array_equal(again.states, res.states) and again.sweeps.max() == 1


def test_retrieves_from_noise_at_low_load():
    r = rng(10)
    X = random_patterns(6, 120, r)
    Q = flip_noise(X, 0.15, r)
    out = C.retrieve(C.hebbian(X), Q, rng=r).states
    assert np.array_equal(out, X)


def test_retrieve_is_deterministic_given_rng():
    X = random_patterns(10, 60, rng(11))
    Q = flip_noise(X, 0.2, rng(12))
    a = C.retrieve(C.storkey(X), Q, rng=rng(13)).states
    b = C.retrieve(C.storkey(X), Q, rng=rng(13)).states
    assert np.array_equal(a, b)


def test_flip_noise_exact_count():
    x = random_patterns(4, 50, rng())
    y = flip_noise(x, 0.2, rng(1))
    assert np.all((x != y).sum(axis=1) == 10)


def test_correlated_patterns_overlap_grows_with_rho():
    n = 4000
    lo = correlated_patterns(2, n, 0.2, rng(0))
    hi = correlated_patterns(2, n, 0.8, rng(0))
    assert abs(lo[0] @ lo[1] / n) < abs(hi[0] @ hi[1] / n)
    assert abs(hi[0] @ hi[1] / n - 0.64) < 0.08
    with pytest.raises(ValueError):
        correlated_patterns(2, 5, 1.0, rng())


def test_wrong_update_rule_does_raise_energy():
    """Sanity: the energy test above has teeth. Updating *against* the field must raise the energy."""
    r = rng(14)
    X = random_patterns(5, 30, r)
    W = C.hebbian(X)
    s = X[0].copy()
    e0 = C.energy(W, s)[0]
    for i in range(3):
        s[i] = -np.sign(W[i] @ s) or s[i]
    assert C.energy(W, s)[0] > e0
