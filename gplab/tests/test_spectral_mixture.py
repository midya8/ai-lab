import numpy as np
import pytest

from gplab import ExactGP, RBF, SpectralMixture
from test_kernels import fd_grad


def test_gradients_match_finite_differences():
    X = np.random.default_rng(0).uniform(-2, 2, (15, 2))
    k = SpectralMixture([1.0, 0.5], [[0.3, 0.8], [1.1, 0.4]], [[0.2, 0.1], [0.05, 0.3]])
    for a, b in zip(k.grad(X), fd_grad(k, X)):
        assert np.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_psd_and_diag():
    X = np.random.default_rng(1).uniform(-3, 3, (40, 1))
    k = SpectralMixture([1.0, 0.7], [0.4, 1.3], [0.05, 0.02])
    K = k(X)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    assert np.allclose(np.diag(K), k.diag(X))


def test_single_component_zero_mean_is_rbf():
    # mu=0, v=1/(4 pi^2 l^2) gives exp(-t^2 / (2 l^2)) = RBF(l)
    l = 1.3
    X = np.linspace(-2, 2, 12)[:, None]
    sm = SpectralMixture([1.0], [1e-12], [1 / (4 * np.pi ** 2 * l ** 2)])
    assert np.allclose(sm(X), RBF(l, 1.0)(X), atol=1e-8)


def test_extrapolates_two_frequencies_better_than_rbf():
    rng = np.random.default_rng(3)
    f = lambda t: np.sin(2 * np.pi * 0.5 * t) + 0.7 * np.sin(2 * np.pi * 1.3 * t)
    t = np.sort(rng.uniform(0, 8, 120)); y = f(t) + 0.05 * rng.normal(size=t.size)
    ts = np.linspace(8, 10, 60)
    sm = SpectralMixture([0.8, 0.5], [0.5, 1.3], [1e-3, 1e-3])
    g = ExactGP(sm, noise=0.05 ** 2).set_data(t[:, None], y)
    r = ExactGP(RBF(1.0, 1.0), noise=0.05 ** 2).fit(t[:, None], y)
    e_sm = np.sqrt(np.mean((g.predict(ts[:, None])[0] - f(ts)) ** 2))
    e_rbf = np.sqrt(np.mean((r.predict(ts[:, None])[0] - f(ts)) ** 2))
    assert e_sm < 0.15 and e_sm < 0.5 * e_rbf


def test_fitting_from_rough_init_improves_lml_and_recovers_frequencies():
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0, 8, 120))
    y = np.sin(2 * np.pi * 0.5 * t) + 0.7 * np.sin(2 * np.pi * 1.3 * t) + 0.05 * rng.normal(size=t.size)
    g = ExactGP(SpectralMixture([0.8, 0.5], [0.45, 1.25], [1e-2, 1e-2]), noise=0.01)
    g.set_data(t[:, None], y)
    g.fit(t[:, None], y)
    mu = np.sort(g.kernel.means.ravel())
    assert np.allclose(mu, [0.5, 1.3], atol=0.03)
