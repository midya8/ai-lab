import numpy as np
import pytest

from gplab import RBF, Linear, Matern32, Matern52, Periodic


def fd_grad(k, X, eps=1e-6):
    th = k.params.copy()
    out = []
    for i in range(th.size):
        e = np.zeros_like(th); e[i] = eps
        k.params = th + e; Kp = k(X)
        k.params = th - e; Km = k(X)
        k.params = th
        out.append((Kp - Km) / (2 * eps))
    return out


KERNELS = [
    lambda: RBF(1.2, 0.8),
    lambda: RBF([0.7, 1.5], 2.0),
    lambda: Matern32(0.9, 1.1, ard_dims=2),
    lambda: Matern52([1.0, 2.0], 0.5),
    lambda: Periodic(0.8, 1.7, 1.2),
    lambda: Linear(0.7),
    lambda: RBF(1.0, 1.0) * Periodic(1.0, 2.0, 1.0) + Linear(0.3),
    lambda: (Matern32(1.0, 1.0, ard_dims=2) + RBF([1.0, 2.0], 0.5)) * Matern52(1.5, 2.0, ard_dims=2),
]


@pytest.mark.parametrize("make", KERNELS)
def test_gradient_matches_finite_differences(make):
    X = np.random.default_rng(1).uniform(-3, 3, (20, 2))
    k = make()
    for a, b in zip(k.grad(X), fd_grad(k, X)):
        assert np.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_gradient_check_detects_a_wrong_gradient():
    X = np.random.default_rng(1).uniform(-3, 3, (20, 2))
    k = Matern52(1.0, 1.0, ard_dims=2)
    wrong = [g * 1.01 for g in k.grad(X)]
    assert any(not np.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(wrong, fd_grad(k, X)))


@pytest.mark.parametrize("make", KERNELS)
def test_symmetric_psd_and_diag(make):
    X = np.random.default_rng(2).uniform(-2, 2, (30, 2))
    k = make()
    K = k(X)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    assert np.allclose(np.diag(K), k.diag(X))
    # cross-covariance consistent with the square matrix
    assert np.allclose(k(X, X[:5]), K[:, :5])


def test_known_values():
    X = np.array([[0.0], [1.0]])
    assert np.isclose(RBF(1.0, 2.0)(X)[0, 1], 2.0 * np.exp(-0.5))
    r = np.sqrt(3.0)
    assert np.isclose(Matern32(1.0, 1.0)(X)[0, 1], (1 + r) * np.exp(-r))
    r = np.sqrt(5.0)
    assert np.isclose(Matern52(1.0, 1.0)(X)[0, 1], (1 + r + 5 / 3) * np.exp(-r))
    # periodic kernel repeats exactly at the period
    P = Periodic(0.7, 2.0, 1.5)
    assert np.isclose(P(np.array([[0.0]]), np.array([[2.0]]))[0, 0], 1.5)
    assert np.isclose(Linear(2.0)(np.array([[3.0]]), np.array([[4.0]]))[0, 0], 24.0)


def test_ard_lengthscale_switches_off_a_dimension():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    X2 = X.copy(); X2[:, 1] += rng.normal(size=10)
    k = RBF([1.0, 1e6], 1.0)
    assert np.allclose(k(X), k(X2), atol=1e-8)


def test_params_roundtrip_and_composite_layout():
    k = RBF(1.0, 1.0) * Periodic(1.0, 2.0, 1.0) + Linear(0.3)
    assert k.n_params == 2 + 3 + 1 == k.params.size
    th = np.linspace(-0.5, 0.5, k.n_params)
    k.params = th
    assert np.allclose(k.params, th)
    assert len(k.param_names()) == k.n_params
    k2 = k.copy(); k2.params = th + 1
    assert np.allclose(k.params, th)       # deep copy
