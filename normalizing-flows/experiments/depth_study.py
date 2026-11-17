"""Depth x coupling-type study on datasets with an exact density (so the gap is a KL estimate).

usage: python depth_study.py ring8|checker4
"""
import sys, time
sys.path.insert(0, "../src")
import numpy as np
from normflow import DATASETS, Flow, bridge_mass, fit, nll_gap

name = sys.argv[1]
ds = DATASETS[name]()
SEEDS, STEPS, HID = [0, 1, 2], 3000, 64
DEPTHS = [2, 4, 8, 16]
print(f"# dataset={ds.name} steps={STEPS} batch=256 hidden={HID} train=20000 held-out=20000 seeds={SEEDS}")
print("# true NLL is the exact data entropy (Monte-Carlo, held-out samples); gap = flow NLL - true NLL (nats, KL estimate)")
print(f"{'coupling':9s} {'L':>3s} {'params':>7s} {'gap':>14s} {'bridge%':>12s} {'sec':>5s}")
for additive in (False, True):
    for L in DEPTHS:
        gaps, bridges, t0 = [], [], time.time()
        for s in SEEDS:
            x = ds.sample(20000, 100 + s)
            f = Flow(2, L, HID, additive=additive, seed=s)
            fit(f, x, steps=STEPS, seed=s)
            gaps.append(nll_gap(f, ds)["gap"])
            if ds.centers is not None:
                bridges.append(100 * bridge_mass(f.sample(20000, seed=s), ds.centers, ds.sigma))
        b = f"{np.mean(bridges):5.1f}±{np.std(bridges):4.1f}" if bridges else "     n/a"
        print(f"{'additive' if additive else 'affine':9s} {L:3d} {f.n_params():7d} {np.mean(gaps):7.3f}±{np.std(gaps):5.3f} {b:>12s} {time.time()-t0:5.0f}", flush=True)
