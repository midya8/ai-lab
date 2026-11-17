import numpy as np
from tangent_kernel import ntk
from tangent_kernel.ridge import loo_curve, select_ridge, ridge_predict, ridge_study


def make(n=15, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    X = np.stack([np.cos(phi), np.sin(phi)], 1)
    return X, np.sin(phi) + 0.2 * rng.normal(size=n)


def test_loo_matches_explicit_refit():
    X, y = make()
    K = ntk(X)
    lams = [1e-3, 0.1, 3.0]
    loo, _ = loo_curve(K, y, lams)
    for lam, got in zip(lams, loo):
        errs = []
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            p = ridge_predict(K[np.ix_(m, m)], K[i:i + 1, m], y[m], lam)[0]
            errs.append((p - y[i]) ** 2)
        assert np.isclose(got, np.mean(errs), rtol=1e-6)


def test_wrong_hat_diag_would_fail():
    X, y = make()
    K = ntk(X)
    loo, tr = loo_curve(K, y, [0.1])
    assert loo[0] > tr[0]          # LOO error strictly exceeds train error


def test_loo_beats_interpolation_on_noisy_data():
    out = ridge_study(n=30, noise=0.4, seeds=6)
    assert out["loo"] < out["interp"]
    assert out["oracle"] <= out["loo"] + 1e-12


def test_select_ridge_in_grid():
    X, y = make()
    lams = np.logspace(-4, 2, 10)
    assert select_ridge(ntk(X), y, lams) in lams
