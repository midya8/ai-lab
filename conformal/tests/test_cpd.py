import numpy as np
from conformal_pred import *


def _fit(seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr = hetero_regression(1500, rng)
    Xc, yc = hetero_regression(2000, rng)
    Xt, yt = hetero_regression(20000, rng)
    mean = RidgeRegressor(RFF(1, 120, 0.7, seed=0), 0.1).fit(Xtr, ytr)
    return mean, Xc, yc, Xt, yt


def test_randomised_pit_is_uniform():
    mean, Xc, yc, Xt, yt = _fit()
    cpd = ConformalPredictiveDistribution(mean.predict, lambda X: sigma_true(X[:, 0]), randomised=True).calibrate(Xc, yc)
    u = cpd.cdf(Xt, yt)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert abs(np.mean(u <= p) - p) < 0.02


def test_cdf_monotone_and_quantile_inverts():
    mean, Xc, yc, Xt, yt = _fit(1)
    cpd = ConformalPredictiveDistribution(mean.predict, lambda X: sigma_true(X[:, 0])).calibrate(Xc, yc)
    X = Xt[:50]
    ys = np.linspace(-4, 4, 40)
    F = np.array([cpd.cdf(X, y) for y in ys])
    assert np.all(np.diff(F, axis=0) >= 0)
    q = cpd.quantile(X, 0.7)
    assert np.all(np.abs(cpd.cdf(X, q) - 0.7) < 0.01)


def test_interval_coverage_and_extremes():
    mean, Xc, yc, Xt, yt = _fit(2)
    cpd = ConformalPredictiveDistribution(mean.predict, lambda X: sigma_true(X[:, 0])).calibrate(Xc, yc)
    lo, hi = cpd.interval(Xt, 0.1)
    assert abs(interval_coverage(lo, hi, yt) - 0.9) < 0.02
    assert cpd.quantile(Xt[:3], 1 - 1e-9).max() == np.inf


def test_crps_prefers_correct_scale():
    mean, Xc, yc, Xt, yt = _fit(3)
    good = ConformalPredictiveDistribution(mean.predict, lambda X: sigma_true(X[:, 0])).calibrate(Xc, yc)
    flat = ConformalPredictiveDistribution(mean.predict).calibrate(Xc, yc)
    assert good.crps(Xt[:300], yt[:300]) < flat.crps(Xt[:300], yt[:300])
