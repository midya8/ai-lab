"""Covariate-shift study: train/calibrate on x~U[-3,3], test on x~N(1.5,0.7) truncated to [-3,3]
(toward the noisy right edge). Constant-width split conformal vs weighted conformal with the exact
density ratio and with a logistic-estimated one. 5 seeds, hyperparameters as in run_all.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from conformlab import *

A, MU, SD = 0.1, 1.5, 0.7
out = []
P = lambda s="": (print(s), out.append(s))


def shifted(n, rng):
    x = rng.normal(MU, SD, 4 * n)
    x = x[np.abs(x) <= 3][:n]
    return x[:, None], mean_true(x) + sigma_true(x) * rng.normal(size=len(x))


exact_w = lambda X: np.exp(-0.5 * ((X[:, 0] - MU) / SD) ** 2)   # up to a constant: uniform cal density is flat
agg = {}
for seed in range(5):
    rng = np.random.default_rng(seed)
    Xtr, ytr = hetero_regression(2000, rng)
    Xc, yc = hetero_regression(1000, rng)
    Xt, yt = shifted(20000, rng)
    Xu, _ = shifted(1000, rng)                       # unlabeled test-domain sample for the estimator
    feat = RFF(1, 120, 0.7, seed=seed)
    mean = RidgeRegressor(feat, 0.1).fit(Xtr, ytr)
    ms = {"unweighted (ignores shift)": SplitConformal(mean.predict, None, A),
          "weighted, exact ratio": WeightedSplitConformal(mean.predict, exact_w, None, A),
          "weighted, logistic-estimated": WeightedSplitConformal(mean.predict, logistic_density_ratio(Xc, Xu), None, A)}
    for k, m in ms.items():
        m.calibrate(Xc, yc)
        l, h = m.predict(Xt)
        agg.setdefault(k, []).append((((yt >= l) & (yt <= h)).mean(), np.mean(np.minimum(h - l, 1e6)), np.isinf(h).mean()))
P(f"# shift study (alpha={A}, cal n=1000, test n=20000, mean over 5 seeds)")
P(f"{'method':30s} {'coverage':>8s} {'width':>7s} {'inf-frac':>8s}")
for k, v in agg.items():
    a = np.mean(v, 0)
    P(f"{k:30s} {a[0]:8.3f} {a[1]:7.2f} {a[2]:8.3f}")
open(os.path.join(os.path.dirname(__file__), "shift_results.txt"), "w").write("\n".join(out) + "\n")
