import numpy as np
import pytest

from gplab import RBF, ExactGP, RobustGP


def make(seed=0, n=60, n_out=6):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, (n, 1)), axis=0)
    f = np.sin(X[:, 0])
    y = f + 0.1 * rng.standard_normal(n)
    idx = rng.choice(n, n_out, replace=False)
    y[idx] += rng.choice([-1, 1], n_out) * rng.uniform(2, 4, n_out)
    return X, y, f, idx


def test_robust_beats_gaussian_under_outliers():
    for seed in range(3):
        X, y, f, _ = make(seed)
        g = ExactGP(RBF(1.0, 1.0), 0.01).fit(X, y)
        g.log_noise = np.log(0.01)
        g.kernel.params = RBF(1.0, 1.0).params
        g.set_data(X, y)
        r = RobustGP(RBF(1.0, 1.0), noise=0.01, nu=3).fit(X, y)
        eg = np.sqrt(np.mean((g.predict(X)[0] - f) ** 2))
        er = np.sqrt(np.mean((r.predict(X)[0] - f) ** 2))
        assert er < 0.6 * eg


def test_outliers_get_small_weights():
    X, y, _, idx = make(1)
    r = RobustGP(RBF(1.0, 1.0), noise=0.01, nu=3).fit(X, y)
    assert set(idx) <= set(r.outliers(0.3))
    clean = np.setdiff1d(np.arange(len(y)), idx)
    assert np.median(r.weights_[clean]) > 5 * r.weights_[idx].max()


def test_matches_gaussian_gp_when_nu_huge_and_clean():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (30, 1)); y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(30)
    r = RobustGP(RBF(1.0, 1.0), noise=0.01, nu=1e6).fit(X, y)
    g = ExactGP(RBF(1.0, 1.0), 0.01).set_data(X, y)
    Xs = np.linspace(-3, 3, 20)[:, None]
    assert np.allclose(r.predict(Xs)[0], g.predict(Xs)[0], atol=1e-3)
    assert np.allclose(r.weights_, 1.0, atol=1e-3)


def test_validation():
    with pytest.raises(ValueError):
        RobustGP(RBF(1.0, 1.0), nu=0)
