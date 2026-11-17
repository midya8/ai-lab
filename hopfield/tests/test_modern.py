import numpy as np

from hopfield import modern as M
from hopfield import classical as C
from hopfield.patterns import random_patterns, flip_noise


def rng(s=0):
    return np.random.default_rng(s)


def unit(p, d, seed=0):
    X = rng(seed).normal(size=(p, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_softmax_rows_sum_to_one_and_stable():
    z = np.array([[1000.0, 1001.0, 999.0]])
    p = M.softmax(z)
    assert np.isfinite(p).all() and np.allclose(p.sum(), 1.0)


def test_update_equals_softmax_attention():
    X = unit(20, 16, 1)
    Q = rng(2).normal(size=(5, 16))
    d = 16
    new, _ = M.cont_update(X, Q, beta=1.0 / np.sqrt(d))
    assert np.allclose(new, M.attention(Q, X, X))
    # explicit formula, row by row
    for q, out in zip(Q, new):
        s = X @ q / np.sqrt(d)
        w = np.exp(s - s.max()); w /= w.sum()
        assert np.allclose(out, w @ X)


def test_energy_decreases_monotonically_under_updates():
    X = unit(30, 24, 3)
    Q = rng(4).normal(size=(8, 24)) * 0.5
    for beta in (0.5, 4.0, 15.0):
        res = M.cont_retrieve(X, Q, beta, max_steps=60)
        assert np.all(np.diff(res.energies, axis=0) <= 1e-10), beta


def test_fixed_point_at_high_beta_is_stored_pattern():
    X = unit(16, 64, 5)
    Q = X + rng(6).normal(size=X.shape) * 0.03
    res = M.cont_retrieve(X, Q, beta=40.0)
    assert np.allclose(res.states, X, atol=1e-6)
    assert np.allclose(M.effective_number(res.weights), 1.0, atol=1e-3)


def test_low_beta_gives_global_mean():
    X = unit(16, 64, 7)
    res = M.cont_retrieve(X, X[:3], beta=1e-3)
    assert np.allclose(res.states, X.mean(axis=0), atol=1e-2)
    assert np.all(M.effective_number(res.weights) > 15.9)


def test_effective_number_extremes():
    assert np.isclose(M.effective_number(np.ones((1, 8)) / 8)[0], 8.0)
    assert np.isclose(M.effective_number(np.eye(5)[:1])[0], 1.0)


def test_separation_values():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    d = M.separation(X)
    assert np.allclose(d, [1 - 0.6, 1 - 0.8, 1 - 0.8])


def test_one_step_error_shrinks_exponentially_with_separation():
    X = unit(32, 64, 8)
    err = []
    for beta in (8.0, 16.0, 32.0):
        new, _ = M.cont_update(X, X, beta)
        err.append(np.max(np.linalg.norm(new - X, axis=1)))
    assert err[0] > err[1] > err[2]
    assert err[2] < 1e-3


def test_dam_energy_non_increasing_per_sweep_exp_and_poly():
    r = rng(9)
    X = random_patterns(30, 60, r)
    Q = flip_noise(X[:6], 0.3, r)
    for kw in (dict(kind="exp", beta=6.0), dict(kind="poly", order=3), dict(kind="poly", order=2)):
        res = M.dam_retrieve(X, Q, rng=r, **kw)
        assert np.all(np.diff(res.energies, axis=0) <= 1e-12), kw
        assert res.converged.all()


def test_dam_matches_bruteforce_energy_descent_rule():
    r = rng(10)
    X = random_patterns(7, 20, r)
    s = r.choice([-1.0, 1.0], size=20)
    Q = s[None, :]
    out = M.dam_retrieve(X, Q, kind="exp", beta=5.0, max_sweeps=1, rng=rng(1)).states[0]
    # replay the same permutation by hand, choosing the lower-energy sign by brute force
    t = s.copy()
    for i in rng(1).permutation(20):
        up, dn = t.copy(), t.copy()
        up[i], dn[i] = 1.0, -1.0
        eu = M.dam_energy(X, up[None], "exp", beta=5.0)[0]
        ed = M.dam_energy(X, dn[None], "exp", beta=5.0)[0]
        if eu < ed - 1e-12:
            t[i] = 1.0
        elif ed < eu - 1e-12:
            t[i] = -1.0
    assert np.array_equal(out, t)


def test_exponential_dam_stores_far_beyond_hebbian():
    r = rng(11)
    n, p = 100, 400                                   # alpha = 4, Hebbian capacity is 0.138
    X = random_patterns(p, n, r)
    idx = r.choice(p, size=20, replace=False)
    Q = flip_noise(X[idx], 0.15, r)
    out = M.dam_retrieve(X, Q, kind="exp", beta=12.0, rng=r).states
    assert np.array_equal(out, X[idx])
    hb = C.retrieve(C.hebbian(X), Q, rng=r).states
    assert (hb == X[idx]).all(axis=1).mean() < 0.05


def test_higher_order_polynomial_has_more_capacity():
    r = rng(12)
    n, p = 100, 400
    X = random_patterns(p, n, r)
    idx = r.choice(p, size=20, replace=False)
    Q = flip_noise(X[idx], 0.25, r)
    acc = {o: (M.dam_retrieve(X, Q, kind="poly", order=o, rng=rng(0)).states == X[idx]).all(axis=1).mean() for o in (2, 5)}
    assert acc[5] > acc[2]
