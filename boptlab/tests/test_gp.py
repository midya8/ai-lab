import numpy as np
import pytest
from boptlab import GP, Kernel, fit_gp, log_marginal_likelihood
from boptlab.gp import _chol


def data(n=12, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_kernel_is_psd_with_variance_on_diagonal(kind):
    X = np.random.default_rng(1).random((30, 3))
    k = Kernel([0.3, 0.5, 1.0], 2.5, kind)
    K = k(X, X)
    assert np.allclose(np.diag(K), 2.5)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_posterior_matches_brute_force_inverse(kind):
    X, y = data()
    Xs = np.random.default_rng(5).random((7, 2))
    k = Kernel([0.4, 0.7], 1.3, kind)
    gp = GP(k, sn2=1e-2, standardise=False).fit(X, y)
    mu, cov = gp.predict(Xs, return_cov=True)
    Kinv = np.linalg.inv(k(X, X) + 1e-2 * np.eye(len(X)))
    Ks = k(Xs, X)
    assert np.allclose(mu, Ks @ Kinv @ y, atol=1e-8)
    assert np.allclose(cov, k(Xs, Xs) - Ks @ Kinv @ Ks.T, atol=1e-8)
    _, sd = gp.predict(Xs)
    assert np.allclose(sd**2, np.diag(cov), atol=1e-8)


def test_noise_free_gp_interpolates_and_prior_far_away():
    X, y = data()
    gp = GP(Kernel([0.3, 0.3], 1.0), sn2=1e-8, standardise=False).fit(X, y)
    mu, sd = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3) and sd.max() < 1e-2
    mu_far, sd_far = gp.predict(np.array([[50.0, 50.0]]))
    assert abs(mu_far[0]) < 1e-6 and abs(sd_far[0] - 1.0) < 1e-6


def test_standardisation_is_transparent_on_original_scale():
    X, y = data()
    gp = GP(Kernel([0.4, 0.4], 1.0), 1e-4).fit(X, 100 + 7 * y)
    mu, _ = gp.predict(X)
    assert np.allclose(mu, 100 + 7 * y, atol=0.5)


def test_loglik_matches_dense_gaussian_logpdf():
    X, y = data()
    k = Kernel([0.5, 0.9], 1.7, "matern52")
    K = k(X, X) + 0.05 * np.eye(len(X))
    sign, logdet = np.linalg.slogdet(K)
    ref = -0.5 * y @ np.linalg.solve(K, y) - 0.5 * logdet - 0.5 * len(y) * np.log(2 * np.pi)
    assert sign > 0
    assert log_marginal_likelihood(X, y, k, 0.05) == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_marginal_likelihood_gradient_matches_finite_differences(kind):
    X, y = data(n=10, d=3, seed=2)
    y = (y - y.mean()) / y.std()
    theta = np.array([np.log(0.4), np.log(0.8), np.log(1.5), np.log(1.2), np.log(0.05)])

    def f(th):
        return log_marginal_likelihood(X, y, Kernel(np.exp(th[:3]), np.exp(th[3]), kind), np.exp(th[4]))

    _, g = log_marginal_likelihood(X, y, Kernel(np.exp(theta[:3]), np.exp(theta[3]), kind), np.exp(theta[4]), grad=True)
    fd = np.array([(f(theta + e) - f(theta - e)) / 2e-6 for e in np.eye(5) * 1e-6])
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_matern_gradient_finite_at_coincident_points():
    X = np.vstack([np.random.default_rng(0).random((4, 2))] * 2)
    _, dls, _ = Kernel([0.5, 0.5], 1.0, "matern52").grads(X)
    assert all(np.isfinite(d).all() for d in dls)


def test_fit_improves_likelihood_and_recovers_lengthscale_roughly():
    rng = np.random.default_rng(3)
    X = rng.random((60, 1))
    true = Kernel([0.15], 1.0, "rbf")
    K = true(X, X) + 1e-6 * np.eye(60)
    y = np.linalg.cholesky(K) @ rng.standard_normal(60) + 0.05 * rng.standard_normal(60)
    gp = fit_gp(X, y, kind="rbf", rng=np.random.default_rng(0))
    ys = (y - y.mean()) / y.std()
    default = log_marginal_likelihood(X, ys, Kernel([0.3], 1.0, "rbf"), 1e-2)
    assert gp.log_marglik > default
    assert 0.08 < gp.kernel.ls[0] < 0.35


def test_posterior_samples_have_posterior_moments():
    X, y = data()
    gp = GP(Kernel([0.4, 0.4], 1.0), 1e-3).fit(X, y)
    Xs = np.random.default_rng(9).random((4, 2))
    S = gp.sample(Xs, 20000, np.random.default_rng(1))
    mu, cov = gp.predict(Xs, return_cov=True)
    assert np.allclose(S.mean(0), mu, atol=0.03 * np.sqrt(np.diag(cov)).max() + 0.01)
    assert np.allclose(np.cov(S.T), cov, atol=0.03 * np.diag(cov).max() + 1e-3)


def test_condition_equals_refit_with_extra_data():
    X, y = data()
    gp = GP(Kernel([0.4, 0.4], 1.0), 1e-3).fit(X[:8], y[:8])
    a = gp.condition(X[8:], y[8:])
    b = GP(Kernel([0.4, 0.4], 1.0), 1e-3).fit(X, y)
    Xs = np.random.default_rng(0).random((5, 2))
    assert np.allclose(a.predict(Xs)[0], b.predict(Xs)[0])


def test_chol_adds_jitter_for_singular_matrix():
    K = np.ones((4, 4))
    L = _chol(K)
    assert np.isfinite(L).all()


def test_bad_kernel_name_rejected():
    with pytest.raises(ValueError):
        Kernel([1.0], 1.0, "periodic")
