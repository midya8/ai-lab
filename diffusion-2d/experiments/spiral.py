"""Spiral study: no analytic score, so judge by sliced-W1 / MMD to fresh data only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from diffusion2d import EpsNet, linear_schedule, net_eps_fn, sample, sliced_wasserstein, mmd_rbf, train
from diffusion2d.shapes import spiral

s = linear_schedule(100)
net = EpsNet(2, 128, 3, 0, T=100, seed=0)
train(net, lambda n, r: (spiral(n, r)[0], None), s, steps=4000, seed=0, p_uncond=0.0)
ref = spiral(6000, np.random.default_rng(10_000))[0]
floor = [sliced_wasserstein(spiral(6000, np.random.default_rng(i))[0], ref) for i in range(5)]
print(f"finite-sample floor sliced-W1: {np.mean(floor):.4f} +- {np.std(floor):.4f}")
print(f"pure noise sliced-W1: {sliced_wasserstein(np.random.default_rng(3).standard_normal((6000,2)), ref):.4f}")
for steps in (5, 10, 25, 50, 100):
    for eta in (0.0, 1.0):
        x = sample(net_eps_fn(net), s, 6000, 2, steps, eta, np.random.default_rng(1))
        print(f"steps={steps:3d} eta={eta:.0f} sW1={sliced_wasserstein(x, ref):.4f} mmd={mmd_rbf(x, ref):.5f}")
