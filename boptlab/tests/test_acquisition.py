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
