import math
import numpy as np
import pytest
from conformlab import *
from conformlab.conformal import conformal_quantile


def test_quantile_rank_and_inf():
    s = np.arange(1, 10.0)           # n=9, alpha=.1 -> k=ceil(10*.9)=9 -> max
    assert conformal_quantile(s, 0.1) == 9.0
    assert conformal_quantile(s[:8], 0.1) == np.inf          # k=ceil(9*.9)=9 > 8
    assert min_calibration_size(0.1) == 9
    assert conformal_quantile(s, 0.5) == 5.0                 # k=ceil(5)=5


def test_coverage_beta_law_exact():
    """Split-conformal coverage given n calibration points ~ Beta(k, n+1-k): check mean and variance
    of empirical coverage over many calibration draws with continuous scores."""
    rng = np.random.default_rng(0)
    n, alpha = 30, 0.2
    k, l = coverage_beta_params(n, alpha)
    covs = []
    for _ in range(3000):
        q = conformal_quantile(rng.normal(size=n), alpha)
        from math import erf, sqrt
        covs.append(0.5 * (1 + erf(q / sqrt(2))))            # exact P(score <= q) for N(0,1)
    mean, var = k / (k + l), k * l / ((k + l) ** 2 * (k + l + 1))
    assert abs(np.mean(covs) - mean) < 4 * math.sqrt(var / 3000)
    assert abs(np.var(covs) / var - 1) < 0.15


def test_marginal_coverage_regression_all_methods():
    rng = np.random.default_rng(1)
    Xtr, ytr = hetero_regression(1500, rng)
    Xc, yc = hetero_regression(1000, rng)
    Xt, yt = hetero_regression(20000, rng)
    feat = RFF(1, 120, 0.7, seed=0)
    mean = RidgeRegressor(feat, 0.1).fit(Xtr, ytr)
    sp = SplitConformal(mean.predict, alpha=0.1).calibrate(Xc, yc)
    assert abs(interval_coverage(*sp.predict(Xt), yt) - 0.9) < 0.02
    lo = QuantileRegressor(feat, 0.05, steps=600).fit(Xtr, ytr)
    hi = QuantileRegressor(feat, 0.95, steps=600).fit(Xtr, ytr)
    cq = CQR(lo.predict, hi.predict, alpha=0.1).calibrate(Xc, yc)
    assert abs(interval_coverage(*cq.predict(Xt), yt) - 0.9) < 0.02


def test_cqr_has_better_conditional_coverage_than_constant_width():
    rng = np.random.default_rng(2)
    Xtr, ytr = hetero_regression(2000, rng)
    Xc, yc = hetero_regression(1500, rng)
    Xt, yt = hetero_regression(20000, rng)
    feat = RFF(1, 120, 0.7, seed=0)
    mean = RidgeRegressor(feat, 0.1).fit(Xtr, ytr)
    sp = SplitConformal(mean.predict, alpha=0.1).calibrate(Xc, yc)
    lo = QuantileRegressor(feat, 0.05, steps=800).fit(Xtr, ytr)
    hi = QuantileRegressor(feat, 0.95, steps=800).fit(Xtr, ytr)
    cq = CQR(lo.predict, hi.predict, alpha=0.1).calibrate(Xc, yc)
    gap = lambda m: np.abs(binned_coverage(Xt, (yt >= m.predict(Xt)[0]) & (yt <= m.predict(Xt)[1]), 5) - 0.9).max()
    assert gap(cq) < gap(sp)


def test_binned_coverage_counts():
    x = np.arange(10)
    cov = binned_coverage(x, np.array([1] * 5 + [0] * 5), bins=2)
    assert list(cov) == [1.0, 0.0]


def test_lac_threshold_and_sets():
    P = np.array([[.7, .2, .1], [.4, .4, .2], [.1, .1, .8]])
    cc = ConformalClassifier(lambda X: P, alpha=0.5, method="lac")
    cc.q = 0.55                         # include labels with p >= .45
    S = cc.predict_sets(None)
    assert S.tolist() == [[True, False, False], [False, False, False], [False, False, True]]
    assert empty_rate(S) == pytest.approx(1 / 3)


def test_aps_scores_extremes():
    P = np.array([[.6, .3, .1]])
    assert aps_scores(P, [0], np.array([1.0]))[0] == pytest.approx(.6)
    assert aps_scores(P, [0], np.array([0.0]))[0] == pytest.approx(0.0)
    assert aps_scores(P, [2], np.array([1.0]))[0] == pytest.approx(1.0)


def test_aps_set_contains_label_iff_score_below_q():
    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(5), size=200)
    y = rng.integers(0, 5, 200)
    u = rng.uniform(size=200)
    s = aps_scores(P, y, u)
    S = aps_sets(P, 0.6, u)
    assert np.array_equal(S[np.arange(200), y], s <= 0.6)


def _clf_setup(seed=4):
    rng = np.random.default_rng(seed)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(30000, rng)
    clf = SoftmaxRegressor(6, steps=300).fit(Xtr, ytr)
    return clf, (Xc, yc), (Xt, yt)


def test_classification_marginal_coverage_lac_and_aps():
    clf, (Xc, yc), (Xt, yt) = _clf_setup()
    for method in ("lac", "aps"):
        cc = ConformalClassifier(clf.predict_proba, 0.1, method).calibrate(Xc, yc)
        assert abs(set_coverage(cc.predict_sets(Xt), yt) - 0.9) < 0.02, method


def test_mondrian_fixes_class_conditional_coverage():
    clf, (Xc, yc), (Xt, yt) = _clf_setup()
    plain = ConformalClassifier(clf.predict_proba, 0.1, "lac").calibrate(Xc, yc)
    mond = ConformalClassifier(clf.predict_proba, 0.1, "lac", mondrian=True).calibrate(Xc, yc)
    cp = class_conditional_coverage(plain.predict_sets(Xt), yt, 6)
    cm = class_conditional_coverage(mond.predict_sets(Xt), yt, 6)
    assert np.abs(cp - 0.9).max() > 0.04          # marginal guarantee hides class gaps
    assert np.abs(cm - 0.9).max() < 0.03          # per-class guarantee holds


def test_mondrian_aps_runs_and_covers():
    clf, (Xc, yc), (Xt, yt) = _clf_setup()
    m = ConformalClassifier(clf.predict_proba, 0.1, "aps", mondrian=True).calibrate(Xc, yc)
    cc = class_conditional_coverage(m.predict_sets(Xt), yt, 6)
    assert np.abs(cc - 0.9).max() < 0.04


def test_aci_gamma_zero_is_static_and_adaptive_recovers_under_shift():
    rng = np.random.default_rng(5)
    T = 4000
    cal = np.abs(rng.normal(size=500))
    scale = np.where(np.arange(T) < T // 2, 1.0, 2.0)          # residuals double at T/2
    new = np.abs(rng.normal(size=T)) * scale
    f = lambda t: (cal, new[t])
    miss0, traj0 = aci(f, T, 0.1, gamma=0.0)
    assert np.all(traj0 == 0.1)
    assert miss0[T // 2:].mean() > 0.25                         # static: undercovers after shift
    miss, traj = aci(f, T, 0.1, gamma=0.02)
    assert abs(miss.mean() - 0.1) < 0.03                        # long-run miss rate -> alpha
    assert traj[T // 2 + 500:].mean() < 0.1                     # alpha_t lowered to widen


def test_aci_long_run_guarantee_bound():
    """|mean miss - alpha| <= (max(a1,1-a1)+gamma)/(gamma*T) holds deterministically, any stream."""
    rng = np.random.default_rng(6)
    T, gamma, alpha = 1000, 0.05, 0.1
    cal = np.abs(rng.normal(size=200))
    new = np.abs(rng.standard_cauchy(T))
    miss, _ = aci(lambda t: (cal, new[t]), T, alpha, gamma)
    assert abs(miss.mean() - alpha) <= (max(alpha, 1 - alpha) + gamma) / (gamma * T) + 1e-12


def test_weighted_quantile_reduces_to_split_with_equal_weights():
    from conformlab import weighted_conformal_quantile
    rng = np.random.default_rng(1)
    s = rng.normal(size=50)
    for a in (0.1, 0.3):
        assert weighted_conformal_quantile(s, np.ones(50), 1.0, a) == conformal_quantile(s, a)


def test_weighted_quantile_inf_and_heavy_test_weight():
    from conformlab import weighted_conformal_quantile
    s = np.arange(1.0, 11)
    assert weighted_conformal_quantile(s, np.ones(10), 1.0, 0.05) == np.inf
    assert weighted_conformal_quantile(s, np.ones(10), 1000.0, 0.5) == np.inf   # test point dominates


def test_weighted_quantile_follows_weights():
    from conformlab import weighted_conformal_quantile
    s = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([0.01, 0.01, 0.01, 10.0])
    assert weighted_conformal_quantile(s, w, 0.01, 0.5) == 4.0                  # mass sits on the largest score
    w2 = np.array([10.0, 0.01, 0.01, 0.01])
    assert weighted_conformal_quantile(s, w2, 0.01, 0.5) == 1.0


def test_logistic_density_ratio_recovers_gaussian_ratio():
    from conformlab import logistic_density_ratio
    rng = np.random.default_rng(0)
    Xc = rng.normal(0, 1, (4000, 1)); Xt = rng.normal(1, 1, (4000, 1))
    w = logistic_density_ratio(Xc, Xt)
    x = np.linspace(-1, 2, 7)[:, None]
    true = np.exp(x[:, 0] - 0.5)                   # N(1,1)/N(0,1)
    assert np.all(np.abs(np.log(w(x)) - np.log(true)) < 0.35)


def test_weighted_conformal_covers_under_shift():
    from conformlab import WeightedSplitConformal
    rng = np.random.default_rng(3)
    mean_fn = lambda X: mean_true(X[:, 0])
    Xc, yc = hetero_regression(1500, rng)
    x = rng.normal(1.5, 0.7, 40000); x = x[np.abs(x) <= 3][:10000]
    yt = mean_true(x) + sigma_true(x) * rng.normal(size=len(x))
    m = WeightedSplitConformal(mean_fn, lambda X: np.exp(-0.5 * ((X[:, 0] - 1.5) / 0.7) ** 2), None, 0.1).calibrate(Xc, yc)
    l, h = m.predict(x[:, None])
    assert abs(((yt >= l) & (yt <= h)).mean() - 0.9) < 0.02


def test_raps_lam0_equals_aps_and_penalty_monotone():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(6), 50)
    y = rng.integers(0, 6, 50)
    u = rng.uniform(size=50)
    assert np.allclose(raps_scores(P, y, u, lam=0.0), aps_scores(P, y, u))
    assert (raps_scores(P, y, u, lam=0.1, k_reg=2) >= aps_scores(P, y, u) - 1e-12).all()
    # sets are nested as q grows, and contain the label iff score <= q
    q = 0.7
    S = raps_sets(P, q, u, 0.1, 2)
    assert (S[np.arange(50), y] == (raps_scores(P, y, u, 0.1, 2) <= q)).all()
    assert (raps_sets(P, 0.9, u, 0.1, 2) >= S).all()


def test_raps_coverage_and_not_larger_than_aps_on_hard_classes():
    rng = np.random.default_rng(2)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(20000, rng)
    clf = SoftmaxRegressor(6, steps=300).fit(Xtr, ytr)
    sizes = {}
    for m in ("aps", "raps"):
        cc = ConformalClassifier(clf.predict_proba, 0.1, m, seed=0, lam=0.05, k_reg=2).calibrate(Xc, yc)
        S = cc.predict_sets(Xt)
        assert abs(set_coverage(S, yt) - 0.9) < 0.02
        sizes[m] = mean_set_size(S)
    assert sizes["raps"] <= sizes["aps"] + 0.05
    with pytest.raises(ValueError):
        ConformalClassifier(clf.predict_proba, 0.1, "raps", mondrian=True)
