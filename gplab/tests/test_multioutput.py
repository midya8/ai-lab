import numpy as np
from gplab import RBF
from gplab.multioutput import ICMGP


def _data(seed=0, n=40, noise=0.05):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, (n, 1)), 0)
    f = np.sin(X[:, 0])
    Y = np.stack([f, -0.8 * f, 0.5 * f], 1) + noise * rng.standard_normal((n, 3))
    return X, Y, f


def test_B_is_psd():
    m = ICMGP(RBF(), 3, rank=2)
    assert np.linalg.eigvalsh(m.B).min() > 0


def test_fit_decreases_nlml_and_recovers_correlation():
    X, Y, _ = _data()
    m = ICMGP(RBF(), 3, rank=1)
    m._ymean = Y.mean(0); m._yv = (Y - m._ymean).T.ravel(); m.X = X
    before = m.nlml()
    m.fit(X, Y)
    assert m.nlml() < before
    B = m.B
    corr = B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])
    assert corr < -0.9  # output 1 is -0.8 * output 0


def test_prediction_accuracy_on_all_outputs():
    X, Y, _ = _data(n=50)
    Xt = np.linspace(-2.5, 2.5, 60)[:, None]
    m = ICMGP(RBF(), 3, rank=1).fit(X, Y)
    ref = np.stack([np.sin(Xt[:, 0]), -0.8 * np.sin(Xt[:, 0]), 0.5 * np.sin(Xt[:, 0])], 1)
    assert np.sqrt(np.mean((m.predict(Xt) - ref) ** 2)) < 0.1


def test_variance_positive_and_shapes():
    X, Y, _ = _data()
    m = ICMGP(RBF(), 3).fit(X, Y, max_iter=10)
    mu, var = m.predict(np.linspace(-4, 4, 7)[:, None], return_var=True)
    assert mu.shape == var.shape == (7, 3) and (var > 0).all()


def test_single_output_matches_exactgp_mean():
    X, Y, _ = _data()
    m = ICMGP(RBF(), 1, rank=1)
    m.W[:] = 1.0; m.log_kappa[:] = -20; m.log_noise[:] = np.log(0.05 ** 2)
    m.X = X; m._ymean = Y[:, 0].mean(); m._yv = Y[:, 0] - m._ymean
    mu = m.predict(X[:5])[:, 0]
    k = RBF(); L = np.linalg.cholesky(k(X) + 0.05 ** 2 * np.eye(len(X)))
    ref = k(X[:5], X) @ np.linalg.solve(L.T, np.linalg.solve(L, m._yv)) + m._ymean
    assert np.allclose(mu, ref, atol=1e-6)
