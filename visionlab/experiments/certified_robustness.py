"""Certified (IBP) vs empirical (PGD) robustness for standard, FGSM-adversarial and IBP training. 3 seeds.

For each model and radius eps we report
  certified: share of test images *proved* correct for every L-inf perturbation <= eps (a lower bound),
  PGD:       accuracy under a 20-step PGD attack (an upper bound on the true robust accuracy),
so the true robust accuracy lies between the two; the gap is what the bound cannot decide.
Setup chosen from a small exploratory sweep on a different test seed (see README); test seed here is 998.
    PYTHONPATH=src python experiments/certified_robustness.py
"""
import time

import numpy as np

from visionlab.attacks import pgd
from visionlab.certify import certified_accuracy, certified_radius
from visionlab.data import make_dataset
from visionlab.model import ModelConfig, build_cnn
from visionlab.train import train

EPS = (0.01, 0.02, 0.03, 0.05)
SEEDS = (0, 1, 2)
EPOCHS = 16
Xte, yte = make_dataset(800, seed=998)
VARIANTS = {
    "standard": dict(),
    "FGSM adv-trained (0.05)": dict(adv_eps=0.05),
    "IBP-trained (eps 0.03)": dict(ibp_eps=0.03),
}


def pgd_acc(m, X, y, eps, batch=200):
    rng = np.random.default_rng(0)
    hit = 0
    for s in range(0, len(X), batch):
        adv = pgd(m, X[s:s + batch], y[s:s + batch], eps, steps=20, rng=rng)
        hit += int((m.predict(adv) == y[s:s + batch]).sum())
    return hit / len(X)


results = {}
for name, kw in VARIANTS.items():
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        Xtr, ytr = make_dataset(4000, seed=seed * 10 + 1)
        m = build_cnn(ModelConfig(seed=seed))
        train(m, Xtr, ytr, epochs=EPOCHS, seed=seed, log=None, **kw)
        rad = certified_radius(m, Xte[:200], yte[:200])
        rows.append({
            "clean": float((m.predict(Xte) == yte).mean()),
            "cert": [certified_accuracy(m, Xte, yte, e) for e in EPS],
            "naive": [certified_accuracy(m, Xte, yte, e, "naive") for e in EPS],
            "pgd": [pgd_acc(m, Xte, yte, e) for e in EPS],
            "radius_median": float(np.median(rad)),
        })
        print(f"  {name} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    results[name] = rows

print("\nmean over 3 seeds (+- std of clean accuracy); n=800 test images, radius on the first 200")
head = f"{'model':<26}{'clean':>8}" + "".join(f"  cert@{e:<5}" for e in EPS)
print(head)
for name, rows in results.items():
    mean = lambda k: np.mean([r[k] for r in rows], axis=0)
    c = mean("cert")
    print(f"{name:<26}{mean('clean'):>8.3f}" + "".join(f"  {v:>10.3f}" for v in c)
          + f"   (clean std {np.std([r['clean'] for r in rows]):.3f})")
print()
for kind, label in (("naive", "naive interval comparison"), ("pgd", "PGD-20 accuracy (upper bound)")):
    print(label)
    for name, rows in results.items():
        v = np.mean([r[kind] for r in rows], axis=0)
        print(f"{name:<26}{'':>8}" + "".join(f"  {x:>10.3f}" for x in v))
print("\nmedian certified radius (first 200 test images, capped at 0.5)")
for name, rows in results.items():
    print(f"{name:<26}{np.mean([r['radius_median'] for r in rows]):>8.4f}")
