import numpy as np
from conformal_pred import *


def _run(seed, noise="gauss"):
    rng = np.random.default_rng(seed)
    Xtr, ytr = hetero_regression(2000, rng, noise)
    Xc, yc = hetero_regression(2000, rng, noise)
    Xt, yt = hetero_regression(20000, rng, noise)
    mean = RidgeRegressor(RFF(1, 120, 0.7, seed=0), 0.1).fit(Xtr, ytr)
    lwc = LocallyWeightedConformal(mean.predict, RidgeRegressor(RFF(1, 40, 0.7, seed=1), 1.0), alpha=0.1)
    lwc.fit_scale(Xtr, ytr).calibrate(Xc, yc)
    plain = SplitConformal(mean.predict, alpha=0.1).calibrate(Xc, yc)
    return lwc, plain, Xt, yt


def test_marginal_coverage_valid():
    for seed in range(3):
        lwc, _, Xt, yt = _run(seed)
        lo, hi = lwc.predict(Xt)
        assert 0.88 < interval_coverage(lo, hi, yt) < 0.93


def test_valid_even_with_wrong_scale_model():
    lwc, _, Xt, yt = _run(0, "t3")
    lo, hi = lwc.predict(Xt)
    assert interval_coverage(lo, hi, yt) > 0.88


def test_more_adaptive_than_constant_width():
    lwc, plain, Xt, yt = _run(0)
    def spread(m):
        lo, hi = m.predict(Xt)
        cov = (yt >= lo) & (yt <= hi)
        b = np.digitize(Xt[:, 0], np.linspace(-3, 3, 7)[1:-1])
        c = [cov[b == k].mean() for k in range(6)]
        return max(c) - min(c)
    assert spread(lwc) < spread(plain)
    # width varies with x for the learned scale, not for the constant one
    lo, hi = lwc.predict(np.array([[0.0], [2.9]]))
    assert (hi - lo)[1] > 2 * (hi - lo)[0]
