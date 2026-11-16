import numpy as np

from gplab import RBF, ExactGP, Linear, Matern32, coverage, gaussian_nlpd, lbfgs, sample_gp_data


def brute_lml(K, y, s2):
    C = K + s2 * np.eye(len(y))
    sign, logdet = np.linalg.slogdet(C)
    return -0.5 * y @ np.linalg.solve(C, y) - 0.5 * logdet - 0.5 * len(y) * np.log(2 * np.pi)


def test_lml_matches_multivariate_normal_formula(data1d):
    X, y = data1d
    gp = ExactGP(RBF(0.9, 1.3), 0.07).set_data(X, y)
    assert np.isclose(gp.log_marginal_likelihood(), brute_lml(gp.kernel(X), y, 0.07))


def test_lml_gradient_matches_finite_differences(data1d):
    X, y = data1d
    gp = ExactGP(RBF(0.9, 1.3) + Matern32(2.0, 0.4), 0.07).set_data(X, y)
    v, g = gp.lml_and_grad()
    th = gp.params.copy()
    fd = []
    for i in range(th.size):
        e = np.zeros_like(th); e[i] = 1e-6
        gp.params = th + e; a = gp.log_marginal_likelihood()
        gp.params = th - e; b = gp.log_marginal_likelihood()
        fd.append((a - b) / 2e-6)
    gp.params = th
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_predict_matches_dense_formula(data1d):
    X, y = data1d
    gp = ExactGP(RBF(0.9, 1.3), 0.07).set_data(X, y)
    Xs = np.linspace(-3, 3, 11)[:, None]
    K = gp.kernel(X) + 0.07 * np.eye(len(y))
    Ks = gp.kernel(Xs, X)
    mean = Ks @ np.linalg.solve(K, y)
    cov = gp.kernel(Xs) - Ks @ np.linalg.solve(K, Ks.T)
    m, v = gp.predict(Xs)
    assert np.allclose(m, mean) and np.allclose(v, np.diag(cov), atol=1e-9)
    m2, c2 = gp.predict_cov(Xs)
    assert np.allclose(c2, cov, atol=1e-9)
    _, vn = gp.predict(Xs, include_noise=True)
    assert np.allclose(vn, v + 0.07)


def test_low_noise_interpolates_and_far_field_reverts_to_prior():
    X = np.random.default_rng(0).uniform(-3, 3, (40, 1))
    y = np.sin(X[:, 0])
    gp = ExactGP(RBF(1.0, 2.0), 1e-8).set_data(X, y)
    m, v = gp.predict(X)
    assert np.allclose(m, y, atol=1e-3) and v.max() < 1e-3
    m, v = gp.predict(np.array([[500.0]]))
    assert abs(m[0]) < 1e-6 and np.isclose(v[0], 2.0)


def test_closed_form_loo_equals_brute_force_refits(data1d):
    X, y = data1d
    X, y = X[:15], y[:15]
    gp = ExactGP(RBF(0.9, 1.3), 0.07).set_data(X, y)
    mu, var = gp.loo()
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        g = ExactGP(RBF(0.9, 1.3), 0.07).set_data(X[keep], y[keep])
        m, v = g.predict(X[i:i + 1], include_noise=True)
        assert np.isclose(mu[i], m[0], atol=1e-8) and np.isclose(var[i], v[0], atol=1e-8)


def test_centering_equals_shifting_targets(data1d):
    X, y = data1d
    a = ExactGP(RBF(1, 1), 0.1, center=True).set_data(X, y + 5.0)
    b = ExactGP(RBF(1, 1), 0.1, center=True).set_data(X, y)
    Xs = np.linspace(-3, 3, 7)[:, None]
    assert np.allclose(a.predict(Xs)[0], b.predict(Xs)[0] + 5.0)


def test_fit_improves_lml_and_recovers_noise():
    X, y, _ = sample_gp_data(RBF(0.8, 1.5), 150, 0.05, seed=3)
    gp = ExactGP(RBF(1.0, 1.0), 0.5).set_data(X, y)
    before = gp.log_marginal_likelihood()
    gp.fit(X, y, restarts=2)
    assert gp.log_marginal_likelihood() > before + 5
    assert 0.02 < gp.noise < 0.1 and 0.5 < gp.kernel.lengthscale[0] < 1.3
    # the optimum is a stationary point
    assert np.abs(gp.lml_and_grad()[1]).max() < 1e-2


def test_posterior_samples_have_the_posterior_moments(data1d):
    X, y = data1d
    gp = ExactGP(RBF(1.0, 1.0), 0.1).set_data(X, y)
    Xs = np.linspace(-3, 3, 6)[:, None]
    S = gp.sample_posterior(Xs, 4000, seed=0)
    m, c = gp.predict_cov(Xs)
    assert np.allclose(S.mean(0), m, atol=0.05)
    assert np.allclose(np.cov(S.T), c, atol=0.05)


def test_prior_samples_have_the_prior_variance():
    gp = ExactGP(RBF(1.0, 2.0), 0.1)
    S = gp.sample_prior(np.linspace(0, 5, 5)[:, None], 5000, seed=0, include_noise=False)
    assert np.allclose(S.var(0), 2.0, atol=0.15)


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    assert abs(coverage(y, 0.0, 1.0) - 0.95) < 0.01
    assert abs(gaussian_nlpd(y, 0.0, 1.0) - 0.5 * np.log(2 * np.pi * np.e)) < 0.02
    assert gaussian_nlpd(y, 0.0, 0.1) > gaussian_nlpd(y, 0.0, 1.0)   # overconfidence is penalised


def test_correct_model_is_calibrated_and_overconfident_one_is_not():
    X, y, _ = sample_gp_data(RBF(1.0, 1.0), 400, 0.1, seed=0)
    gp = ExactGP(RBF(1.0, 1.0), 0.1).set_data(X[:60], y[:60])
    m, v = gp.predict(X[60:], include_noise=True)
    assert 0.9 < coverage(y[60:], m, v) <= 1.0
    bad = ExactGP(RBF(1.0, 1.0), 1e-4).set_data(X[:60], y[:60])
    m, v = bad.predict(X[60:], include_noise=True)
    assert coverage(y[60:], m, v) < coverage(y[60:], *gp.predict(X[60:], include_noise=True))


def test_lbfgs_on_known_problems():
    def rosen(x):
        f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        g = np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
        return f, g

    x, f, _ = lbfgs(rosen, np.array([-1.2, 1.0]), max_iter=200)
    assert f < 1e-8 and np.allclose(x, [1, 1], atol=1e-3)
    A = np.diag([1.0, 10.0, 100.0])
    x, f, _ = lbfgs(lambda z: (0.5 * z @ A @ z, A @ z), np.ones(3), max_iter=100)
    assert f < 1e-12


def test_lbfgs_respects_bounds():
    x, f, _ = lbfgs(lambda z: (float(np.sum((z - 20.0) ** 2)), 2 * (z - 20.0)), np.zeros(2), bounds=(-9, 9))
    assert np.allclose(x, 9.0)
