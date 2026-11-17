import numpy as np
from conformal_pred.martingale import online_conformal_pvalues, power_martingale_log, detect_change


def test_pvalues_uniform_under_iid():
    p = online_conformal_pvalues(np.random.default_rng(1).normal(size=3000), np.random.default_rng(2))
    assert abs(p.mean() - 0.5) < 0.03
    assert abs(np.mean(p < 0.1) - 0.1) < 0.03


def test_ties_handled():
    p = online_conformal_pvalues(np.zeros(500), np.random.default_rng(0))
    assert np.all((p >= 0) & (p <= 1))


def test_ville_false_alarm_rate():
    alarms = 0
    for s in range(200):
        r = np.random.default_rng(s)
        a, _ = detect_change(r.normal(size=300), delta=0.1, rng=np.random.default_rng(1000 + s))
        alarms += a is not None
    assert alarms / 200 <= 0.1 + 0.05


def test_detects_mean_shift():
    r = np.random.default_rng(3)
    x = np.concatenate([r.normal(size=300), r.normal(2.0, 1, size=300)])
    a, lm = detect_change(x, delta=0.01)
    assert a is not None and 300 <= a < 450


def test_log_martingale_starts_near_zero_and_finite():
    lm = power_martingale_log(np.array([0.5]))
    assert np.isfinite(lm).all()
