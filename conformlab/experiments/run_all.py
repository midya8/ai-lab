"""Regenerates results.txt. 5 seeds, fixed hyperparameters (not tuned on the test numbers)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from conformlab import *
from conformlab.conformal import aci

A = 0.1
out = []
P = lambda s="": (print(s), out.append(s))


def reg(noise, x_dist, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr = hetero_regression(2000, rng, noise, x_dist)
    Xc, yc = hetero_regression(1000, rng, noise, x_dist)
    Xt, yt = hetero_regression(30000, rng, noise, x_dist)
    feat = RFF(1, 120, 0.7, seed=seed)
    mean = RidgeRegressor(feat, 0.1).fit(Xtr, ytr)
    res = np.abs(ytr - mean.predict(Xtr))
    sig = RidgeRegressor(feat, 1.0).fit(Xtr, res)
    lo = QuantileRegressor(feat, A / 2, steps=800).fit(Xtr, ytr)
    hi = QuantileRegressor(feat, 1 - A / 2, steps=800).fit(Xtr, ytr)
    ms = {"constant-width": SplitConformal(mean.predict, None, A),
          "normalised (sigma-hat)": SplitConformal(mean.predict, sig.predict, A),
          "CQR": CQR(lo.predict, hi.predict, A)}
    rows = {}
    for k, m in ms.items():
        m.calibrate(Xc, yc)
        l, h = m.predict(Xt)
        cov = (yt >= l) & (yt <= h)
        b = binned_coverage(Xt, cov, 5)
        rows[k] = (cov.mean(), (h - l).mean(), b.min(), b.max())
    return rows


P("# conformlab results (alpha=0.1, target coverage 0.90; mean over 5 seeds, test n=30000)")
for noise, xd in [("gauss", "uniform"), ("t3", "uniform"), ("gauss", "normal")]:
    P(f"\n## regression: noise={noise}, x~{xd}")
    P(f"{'method':26s} {'coverage':>8s} {'width':>7s} {'worst-bin':>9s} {'best-bin':>8s}   (5 equal-count x-bins)")
    agg = {}
    for s in range(5):
        for k, v in reg(noise, xd, s).items():
            agg.setdefault(k, []).append(v)
    for k, v in agg.items():
        m = np.mean(v, 0)
        P(f"{k:26s} {m[0]:8.3f} {m[1]:7.2f} {m[2]:9.3f} {m[3]:8.3f}")

P("\n## classification: 6 Gaussian classes, 2 intrinsically hard (std 2.5); softmax regression")
P(f"{'method':22s} {'coverage':>8s} {'set size':>8s} {'empty':>6s} {'worst class':>11s} {'hard-class cov':>14s}")
agg = {}
for s in range(5):
    rng = np.random.default_rng(100 + s)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(30000, rng)
    clf = SoftmaxRegressor(6, steps=300).fit(Xtr, ytr)
    acc = (clf.predict_proba(Xt).argmax(1) == yt).mean()
    agg.setdefault("_acc", []).append(acc)
    for name, meth, mond in [("LAC", "lac", False), ("APS (randomised)", "aps", False),
                             ("LAC + Mondrian", "lac", True), ("APS + Mondrian", "aps", True)]:
        cc = ConformalClassifier(clf.predict_proba, A, meth, mond, seed=s).calibrate(Xc, yc)
        S = cc.predict_sets(Xt)
        cls = class_conditional_coverage(S, yt, 6)
        agg.setdefault(name, []).append((set_coverage(S, yt), mean_set_size(S), empty_rate(S),
                                         cls.min(), cls[[4, 5]].mean()))
P(f"(base classifier accuracy {np.mean(agg.pop('_acc')):.3f})")
for k, v in agg.items():
    m = np.mean(v, 0)
    P(f"{k:22s} {m[0]:8.3f} {m[1]:8.2f} {m[2]:6.3f} {m[3]:11.3f} {m[4]:14.3f}")

P("\n## distribution shift, online (residual scale x2 at the midpoint, T=6000, alpha=0.1)")
P(f"{'gamma':>6s} {'miss rate (all)':>15s} {'miss rate after shift':>22s} {'mean alpha_t after':>18s}")
for g in [0.0, 0.005, 0.02, 0.05]:
    r = []
    for s in range(5):
        rng = np.random.default_rng(200 + s)
        T = 6000
        cal = np.abs(rng.normal(size=500))
        new = np.abs(rng.normal(size=T)) * np.where(np.arange(T) < T // 2, 1.0, 2.0)
        miss, tr = aci(lambda t: (cal, new[t]), T, A, g)
        r.append((miss.mean(), miss[T // 2:].mean(), tr[T // 2:].mean()))
    m = np.mean(r, 0)
    P(f"{g:6.3f} {m[0]:15.3f} {m[1]:22.3f} {m[2]:18.3f}")

open(os.path.join(os.path.dirname(__file__), "results.txt"), "w").write("\n".join(out) + "\n")
