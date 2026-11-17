"""Held-out NLL on datasets with no closed-form density, versus depth.  The reference is the best a
single axis-aligned (diagonal) Gaussian, i.e. ActNorm alone, can do."""
import sys, time
sys.path.insert(0, "../src")
import numpy as np
from normflow import DATASETS, Flow, fit

SEEDS, STEPS, HID = [0, 1, 2], 3000, 64
print(f"# steps={STEPS} batch=256 hidden={HID} train=20000 held-out=20000 seeds={SEEDS}; affine couplings; nats/example (lower is better)")
print(f"{'dataset':8s} {'gauss(diag)':>12s} " + " ".join(f"{'L='+str(L):>13s}" for L in (2, 4, 8, 16)))
for name in ("moons", "spiral"):
    ds = DATASETS[name]()
    xv = ds.sample(20000, 10_000)
    row, g = [], []
    for s in SEEDS:
        base = Flow(2, 0, 4, seed=s); base.data_init(ds.sample(20000, 100 + s)); g.append(base.nll(xv))
    for L in (2, 4, 8, 16):
        r = []
        for s in SEEDS:
            f = Flow(2, L, HID, seed=s); fit(f, ds.sample(20000, 100 + s), steps=STEPS, seed=s); r.append(f.nll(xv))
        row.append(f"{np.mean(r):6.3f}±{np.std(r):5.3f}")
    print(f"{name:8s} {np.mean(g):12.3f} " + " ".join(f"{c:>13s}" for c in row), flush=True)
