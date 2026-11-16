"""eps- vs v-prediction x loss weighting, multi-seed.  python experiments/param_study.py (~4 min).

Unconditional ring-GMM; each config trained for STEPS with 4 seeds; scored by
(a) excess eps error vs the exact optimum averaged over t, (b) sliced-W1 of
10-step and 50-step DDIM samples against a fixed reference set.
Configs: (target, weighting) in eps/eps, v/v (plain v-MSE), eps/min_snr, v/min_snr.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from diffusionlab import EpsNet, linear_schedule, net_eps_fn, ring, sample, sliced_wasserstein
from diffusionlab.diagnostics import eps_error_curve
from diffusionlab.vparam import train_param, v_fn_to_eps_fn

SEEDS, STEPS, N, T = [0, 1, 2, 3], 4000, 6000, 100
g, s = ring(), linear_schedule(T)
ref = g.sample(N, np.random.default_rng(12345))
CONFIGS = [("eps", "eps"), ("v", "v"), ("eps", "min_snr"), ("v", "min_snr")]
print(f"ring, T={T}, steps={STEPS}, seeds={SEEDS}, n={N}")
print(f"{'target':>6} {'weight':>7} | {'excess eps err':>15} | {'sW1@10':>8} {'sW1@50':>8}")
for target, w in CONFIGS:
    ex, s10, s50 = [], [], []
    for sd in SEEDS:
        net = EpsNet(2, 128, 3, 0, T=T, seed=sd)
        train_param(net, lambda n, r: (g.sample(n, r), None), s, target=target, weighting=w,
                    steps=STEPS, seed=sd, p_uncond=0.0)
        fn = net_eps_fn(net)
        if target == "v":
            fn = v_fn_to_eps_fn(fn, s)
        # excess error is defined for eps nets; for v nets evaluate through the converted fn
        class W:  # adapter exposing predict() as eps
            def predict(self, x, t, labels, use_ema=True, _fn=fn):
                return _fn(x, int(t[0]), labels)
        ex.append(np.mean([r["excess"] for r in eps_error_curve(W(), g, s, n=4000)]))
        for steps, acc in [(10, s10), (50, s50)]:
            x = sample(fn, s, N, 2, steps, 0.0, np.random.default_rng(1000 + sd))
            acc.append(sliced_wasserstein(x, ref))
    print(f"{target:>6} {w:>7} | {np.mean(ex):8.4f}+-{np.std(ex):.4f} | {np.mean(s10):8.4f} {np.mean(s50):8.4f}", flush=True)
