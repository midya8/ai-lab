import numpy as np
import pytest

from gplab import RBF
from gplab.classify import LaplaceGPC, expected_calibration_error, log_loss, _log_lik, _sigmoid


def moons(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    y = rng.integers(0, 2, n)
    X = np.where(y[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.4 - np.sin(t)])
    return X + 0.2 * rng.standard_normal((n, 2)), y.astype(float)


def test_mode_is_stationary_point():
    X, y = moons(60, 0)
    g = LaplaceGPC(RBF(1.0, 4.0)).set_data(X, y)
    # at the mode: K^-1 f = grad log p(y|f)
    assert np.allclose(g.a, g.y - _sigmoid(g.f_hat), atol=1e-4)


def test_mode_beats_perturbations():
    X, y = moons(50, 1)
    g = LaplaceGPC(RBF(1.0, 4.0)).set_data(X, y)
    Kinv = np.linalg.inv(g.K + 1e-8 * np.eye(50))
    obj = lambda f: -0.5 * f @ Kinv @ f + _log_lik(y, f)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert obj(g.f_hat) >= obj(g.f_hat + 0.05 * rng.standard_normal(50))


def test_laplace_lml_exact_for_gaussian_limit():
    # with a tiny prior variance the posterior is ~Gaussian: LML close to log of the 1-D integral
    X = np.array([[0.0]]); y = np.array([1.0])
    g = LaplaceGPC(RBF(1.0, 0.05)).set_data(X, y)
    s2 = float(g.kernel(X)[0, 0])
    z, w = np.polynomial.hermite_e.hermegauss(100)
    exact = np.log((w / np.sqrt(2 * np.pi) * _sigmoid(np.sqrt(s2) * z)).sum())
    assert abs(g.log_marginal_likelihood() - exact) < 1e-3


def test_probit_close_to_quadrature_and_plugin_is_overconfident():
    X, y = moons(80, 2)
    g = LaplaceGPC(RBF(1.0, 4.0)).set_data(X, y)
    Xs, _ = moons(200, 3)
    q = g.predict_proba_quad(Xs)
    assert np.max(np.abs(g.predict_proba(Xs) - q)) < 0.02
    plug = g.predict_proba(Xs, "plugin")
    assert np.mean(np.abs(plug - 0.5)) >= np.mean(np.abs(q - 0.5))


def test_fit_improves_lml_and_accuracy():
    X, y = moons(120, 4)
    g = LaplaceGPC(RBF(0.1, 0.5))
    before = g.set_data(X, y).log_marginal_likelihood()
    g.fit(X, y, restarts=2)
    assert g.log_marginal_likelihood() > before
    Xs, ys = moons(400, 5)
    p = g.predict_proba(Xs)
    assert np.mean((p > 0.5) == ys) > 0.88
    assert log_loss(p, ys) < 0.45
    assert expected_calibration_error(p, ys) < 0.12


def test_bad_labels_rejected():
    with pytest.raises(ValueError):
        LaplaceGPC(RBF()).set_data(np.zeros((2, 1)), np.array([0, 2]))
