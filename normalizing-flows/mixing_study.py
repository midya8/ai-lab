"""Fixed permutation vs learned invertible linear mixing (Glow), equal depth. Held-out NLL, 3 seeds."""
import sys, numpy as np
sys.path.insert(0, "src")
from normflow import Flow, fit, DATASETS
for name in ["moons", "spiral", "checker4"]:
    ds = DATASETS[name]()
    xt, xv = ds.sample(4000, seed=0), ds.sample(4000, seed=99)
    row = {}
    for mix in ["perm", "linear"]:
        v = []
        for s in range(3):
            f = Flow(2, 6, 48, seed=s, mixing=mix); fit(f, xt, steps=1500, batch=256, seed=s)
            v.append(f.nll(xv))
        row[mix] = (np.mean(v), np.std(v), f.n_params())
    print(f"{name:13s} perm {row['perm'][0]:.3f}±{row['perm'][1]:.3f} ({row['perm'][2]}p)  linear {row['linear'][0]:.3f}±{row['linear'][1]:.3f} ({row['linear'][2]}p)", flush=True)
