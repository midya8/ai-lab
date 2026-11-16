import numpy as np
from conformlab import quantile_tracker


def test_adversarial_bound_holds():
    # scores in [0, 1], arbitrary (non-iid) sequence: blocks of 0s then 1s
    x = np.concatenate([np.zeros(300), np.ones(300), np.random.default_rng(0).uniform(size=400)])
    alpha, eta = 0.1, 0.05
    miss, _ = quantile_tracker(x, alpha, eta, q0=0.5)
    assert abs(miss.mean() - alpha) <= (1 + eta) / (eta * len(x)) + 1e-12


def test_recovers_after_distribution_shift():
    r = np.random.default_rng(1)
    x = np.concatenate([np.abs(r.normal(size=1500)), np.abs(r.normal(size=1500)) * 3])
    miss, q = quantile_tracker(x, 0.1, 0.05, q0=1.0)
    assert abs(miss[2000:].mean() - 0.1) < 0.04  # after the adaptation transient
    assert q[-200:].mean() > 2 * q[1000:1500].mean()


def test_iid_converges_to_true_quantile():
    x = np.random.default_rng(2).uniform(size=20000)
    _, q = quantile_tracker(x, 0.1, 0.01, q0=0.0)
    assert abs(q[-2000:].mean() - 0.9) < 0.05
