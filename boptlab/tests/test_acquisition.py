import numpy as np
import pytest
from boptlab import (expected_improvement, probability_of_improvement, confidence_bound,
                     mc_expected_improvement, GP, Kernel)
from boptlab.acquisition import norm_cdf, mc_batch_expected_improvement


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)
    assert norm_cdf(-8.0) < 1e-14


def test_ei_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sd, best = np.array([0.3, -0.2, 1.5]), np.array([0.5, 0.1, 2.0]), 0.0
    draws = mu + sd * rng.standard_normal((400000, 3))
    mc = mc_expected_improvement(draws, best)
    assert np.allclose(expected_improvement(mu, sd, best), mc, atol=3e-3)


def test_ei_with_xi_matches_shifted_best():
    mu, sd = np.array([0.1, 0.4]), np.array([0.3, 0.2])
    assert np.allclose(expected_improvement(mu, sd, 1.0, xi=0.25), expected_improvement(mu, sd, 0.75))


def test_ei_zero_variance_limits():
    ei = expected_improvement(np.array([2.0, -1.0]), np.array([0.0, 0.0]), 0.0)
    assert ei[0] == 0.0 and ei[1] == pytest.approx(1.0)


def test_ei_nonnegative_monotone_in_mean_and_sigma():
    sd = np.full(50, 0.4)
    mu = np.linspace(-2, 2, 50)
    e = expected_improvement(mu, sd, 0.0)
    assert (e >= 0).all() and (np.diff(e) <= 1e-12).all()      # lower mean -> more EI
    s = np.linspace(0.01, 3, 50)
    e2 = expected_improvement(np.full(50, 0.5), s, 0.0)           # mean worse than best: more sigma -> more EI
    assert (np.diff(e2) >= -1e-12).all()


def test_ei_deep_tail_is_finite_and_nonnegative():
    e = expected_improvement(np.array([40.0]), np.array([1.0]), 0.0)
    assert np.isfinite(e).all() and e[0] >= 0


def test_pi_values_and_degenerate_case():
    p = probability_of_improvement(np.array([0.0, 1.0, -1.0]), np.array([1.0, 1.0, 0.0]), 0.0)
    assert p[0] == pytest.approx(0.5) and p[1] == pytest.approx(0.158655, abs=1e-5) and p[2] == 1.0


def test_confidence_bound_sign_convention():
    s = confidence_bound(np.array([1.0, 2.0]), np.array([0.5, 0.5]), beta=2.0)
    assert np.allclose(s, [0.0, -1.0])               # lower mean preferred, higher sigma preferred
    assert confidence_bound(1.0, 1.0, 3.0) > confidence_bound(1.0, 1.0, 1.0)


def test_mc_ei_from_gp_samples_matches_closed_form():
    rng = np.random.default_rng(1)
    X = rng.random((8, 1)); y = np.sin(6 * X[:, 0])
    gp = GP(Kernel([0.2], 1.0), 1e-4).fit(X, y)
    Xs = np.linspace(0, 1, 5)[:, None]
    S = gp.sample(Xs, 60000, np.random.default_rng(2))
    mu, sd = gp.predict(Xs)
    assert np.allclose(mc_expected_improvement(S, y.min()), expected_improvement(mu, sd, y.min()), atol=0.01)


def test_batch_ei_is_at_least_single_ei_and_monotone_in_batch():
    rng = np.random.default_rng(3)
    S = rng.standard_normal((50000, 3))
    q1 = mc_batch_expected_improvement(S[:, :1], 0.0)
    q3 = mc_batch_expected_improvement(S, 0.0)
    assert q3 > q1
    assert q1 == pytest.approx(expected_improvement(0.0, 1.0, 0.0), abs=0.01)


def test_log_ei_matches_log_of_ei_where_representable():
    from boptlab import log_expected_improvement
    mu = np.linspace(-3, 3, 41); sd = np.full(41, 0.7)
    assert np.allclose(log_expected_improvement(mu, sd, 0.2), np.log(expected_improvement(mu, sd, 0.2)), atol=1e-8)


def test_log_ei_finite_and_monotone_in_deep_tail_and_continuous_at_switch():
    from boptlab import log_expected_improvement
    z = np.linspace(-60, -5, 400)
    v = log_expected_improvement(-z, np.ones_like(z), 0.0)  # imp = z, sigma = 1
    assert np.all(np.isfinite(v)) and np.all(np.diff(v) > 0)
    assert expected_improvement(60.0, 1.0, 0.0) == 0.0  # plain EI has underflowed
    a = log_expected_improvement(10.0 - 1e-9, 1.0, 0.0); b = log_expected_improvement(10.0 + 1e-9, 1.0, 0.0)
    assert abs(a - b) < 1e-6


def test_log_ei_tail_matches_high_precision_reference():
    mpmath = pytest.importorskip("mpmath")
    from boptlab import log_expected_improvement
    mpmath.mp.dps = 60
    for z in [-3.0, -9.9, -10.1, -20.0, -37.5, -80.0]:
        ref = mpmath.log(mpmath.npdf(z) + z * mpmath.ncdf(z))
        got = float(log_expected_improvement(-z, 1.0, 0.0))
        assert got == pytest.approx(float(ref), rel=1e-9, abs=1e-9)


def test_logei_optimizer_runs_and_beats_random():
    from boptlab import BayesOpt, BENCHMARKS
    b = BENCHMARKS["branin"] if isinstance(BENCHMARKS, dict) else BENCHMARKS[0]
    o = BayesOpt(b.bounds, acquisition="logei", rng=np.random.default_rng(0))
    o.run(b.f, 20)
    assert o.y.min() - b.optimum < 1.0


def _fit_gp(kind, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((12, 3))
    y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2 - X[:, 2]
    return GP(Kernel([0.4, 0.6, 0.5], sf2=1.3, kind=kind), sn2=1e-4).fit(X, y), y


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_predict_grad_matches_finite_differences(kind):
    gp, _ = _fit_gp(kind)
    Xs = np.random.default_rng(5).random((6, 3))
    mu, sd, dmu, dsd = gp.predict_grad(Xs)
    m0, s0 = gp.predict(Xs)
    assert np.allclose(mu, m0) and np.allclose(sd, s0)
    h = 1e-6
    for d in range(3):
        e = np.zeros(3); e[d] = h
        mp, sp = gp.predict(Xs + e); mm, sm = gp.predict(Xs - e)
        assert np.allclose(dmu[:, d], (mp - mm) / (2 * h), rtol=1e-4, atol=1e-6)
        assert np.allclose(dsd[:, d], (sp - sm) / (2 * h), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_ei_grad_matches_finite_differences(kind):
    from boptlab import expected_improvement_grad
    gp, y = _fit_gp(kind)
    Xs = np.random.default_rng(7).random((5, 3))
    best = float(y.min())
    ei, g = expected_improvement_grad(gp, Xs, best)
    assert np.allclose(ei, expected_improvement(*gp.predict(Xs), best))
    h = 1e-6
    for d in range(3):
        e = np.zeros(3); e[d] = h
        num = (expected_improvement(*gp.predict(Xs + e), best) - expected_improvement(*gp.predict(Xs - e), best)) / (2 * h)
        assert np.allclose(g[:, d], num, rtol=1e-3, atol=1e-6)
