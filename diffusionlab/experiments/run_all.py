"""Reproduce the README tables: python experiments/run_all.py  (~3 min CPU, 3 seeds).

1. excess noise-prediction error vs the exact optimum, by noise level
2. sampler quality (sliced-W1, on-mode rate) vs number of steps, eta in {0, 1},
   trained net vs exact-score oracle, plus the finite-sample noise floor
3. classifier-free guidance on the overlapping 8-Gaussian ring
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from diffusionlab import (EpsNet, linear_schedule, net_eps_fn, oracle_eps_fn, ring, sample,
                          sliced_wasserstein, mmd_rbf, train)
from diffusionlab.diagnostics import eps_error_curve

SEEDS = [0, 1, 2]
TRAIN_STEPS = 6000
N = 8000
T = 100
g, s = ring(), linear_schedule(T)
ref = g.sample(N, np.random.default_rng(12345))
print(f"ring K={g.K} radius=1.5 std={g.std}; T={T} linear schedule; train steps={TRAIN_STEPS}; n={N}; seeds={SEEDS}")

nets = []
for sd in SEEDS:
    net = EpsNet(2, 128, 3, g.K, T=T, seed=sd)
    train(net, lambda n, r: g.sample(n, r, True), s, steps=TRAIN_STEPS, seed=sd)
    nets.append(net)

print("\n== 1. unconditional excess eps error E|eps_theta - eps*|^2 (mean over seeds; irreducible = what the exact model pays)")
curves = [eps_error_curve(n, g, s, n=8000) for n in nets]
print(f"{'t':>4} {'abar':>8} {'excess':>9} {'(min..max)':>16} {'irreducible':>12}")
for i, r in enumerate(curves[0]):
    ex = [c[i]["excess"] for c in curves]
    print(f"{r['t']:4d} {s.abar[r['t']]:8.4f} {np.mean(ex):9.4f} {min(ex):8.4f}..{max(ex):.4f} {r['irreducible']:12.4f}")

print("\n== 2. sampler quality (sliced-W1 / on-mode %), mean over seeds; oracle uses the exact eps*")
floors = [sliced_wasserstein(g.sample(N, np.random.default_rng(100 + k)), ref) for k in range(5)]
print(f"finite-sample floor (two exact sample sets of {N}): sW1 {np.mean(floors):.4f} (+-{np.std(floors):.4f})")
print(f"{'steps':>5} {'eta':>4} | {'net sW1':>8} {'net on-mode':>11} | {'oracle sW1':>10} {'oracle on-mode':>14}")
for steps, eta in [(100, 1.0), (100, 0.0), (50, 0.0), (20, 0.0), (10, 0.0), (5, 0.0), (20, 1.0), (10, 1.0)]:
    nw, nm, ow, om = [], [], [], []
    for k, net in enumerate(nets):
        rng = np.random.default_rng(1000 + k)
        x = sample(net_eps_fn(net), s, N, 2, steps, eta, rng)
        nw.append(sliced_wasserstein(x, ref)); nm.append(g.mode_stats(x)["on_mode"])
        xo = sample(oracle_eps_fn(g, s), s, N, 2, steps, eta, rng)
        ow.append(sliced_wasserstein(xo, ref)); om.append(g.mode_stats(xo)["on_mode"])
    print(f"{steps:5d} {eta:4.0f} | {np.mean(nw):8.4f} {100*np.mean(nm):10.1f}% | {np.mean(ow):10.4f} {100*np.mean(om):13.1f}%")

print("\n== 3. classifier-free guidance, 50-step DDIM, class-conditional samples (accuracy = nearest mode is the requested class)")
print(f"{'scale':>6} | {'net acc':>8} {'net spread':>10} | {'oracle acc':>10} {'oracle spread':>13}   (true within-class std {g.std})")
for scale in [0.0, 1.0, 2.0, 4.0, 8.0]:
    na, ns, oa, os_ = [], [], [], []
    for k, net in enumerate(nets):
        rng = np.random.default_rng(2000 + k)
        for cls_i in range(g.K):
            lab = np.full(1000, cls_i)
            for fn, acc, spr in [(net_eps_fn(net, scale), na, ns)]:
                x = sample(fn, s, 1000, 2, 50, 0.0, rng, labels=lab)
                acc.append((g.assign(x) == cls_i).mean()); spr.append(x.std(0).mean())
        if k == 0:
            for cls_i in range(g.K):
                lab = np.full(1000, cls_i)
                def ofn(x, t, labels, scale=scale):
                    c = oracle_eps_fn(g, s)(x, t, labels); u = oracle_eps_fn(g, s)(x, t, None)
                    return u + scale * (c - u)
                x = sample(ofn, s, 1000, 2, 50, 0.0, rng, labels=lab)
                oa.append((g.assign(x) == cls_i).mean()); os_.append(x.std(0).mean())
    print(f"{scale:6.1f} | {np.mean(na):8.3f} {np.mean(ns):10.3f} | {np.mean(oa):10.3f} {np.mean(os_):13.3f}")
