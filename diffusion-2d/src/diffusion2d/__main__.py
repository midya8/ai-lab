"""python -m diffusion2d [--steps N] [--seed S]: train on the 8-Gaussian ring and print a report."""
from __future__ import annotations

import argparse

import numpy as np

from . import (EpsNet, linear_schedule, net_eps_fn, oracle_eps_fn, ring, sample,
               sliced_wasserstein, train)
from .diagnostics import ascii_scatter, eps_error_curve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="diffusion2d")
    ap.add_argument("--steps", type=int, default=4000, help="training steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--n", type=int, default=4000, help="samples for metrics")
    a = ap.parse_args(argv)

    g, s = ring(), linear_schedule(a.T)
    net = EpsNet(2, 128, 3, g.K, T=a.T, seed=a.seed)
    print(f"training {net.n_params()} params for {a.steps} steps ...")
    train(net, lambda n, r: g.sample(n, r, True), s, steps=a.steps, seed=a.seed, log_every=max(a.steps // 4, 1))
    ref = g.sample(a.n, np.random.default_rng(10_000))
    print("\nexcess eps error vs exact optimum (t, excess, irreducible):")
    for r in eps_error_curve(net, g, s):
        print(f"  t={r['t']:3d} excess={r['excess']:.4f} irreducible={r['irreducible']:.4f}")
    print("\nsliced-W1 to fresh data (lower is better), net vs exact-score oracle:")
    rng = np.random.default_rng(a.seed + 1)
    for steps, eta in [(a.T, 1.0), (50, 0.0), (20, 0.0), (10, 0.0), (5, 0.0)]:
        xn = sample(net_eps_fn(net, 1.0),
                    s, a.n, 2, steps, eta, rng)
        xo = sample(oracle_eps_fn(g, s), s, a.n, 2, steps, eta, rng)
        print(f"  steps={steps:3d} eta={eta:.0f}  net={sliced_wasserstein(xn, ref):.4f}  "
              f"oracle={sliced_wasserstein(xo, ref):.4f}")
    print("\nunconditional samples (50-step DDIM):")
    print(ascii_scatter(sample(net_eps_fn(net, 1.0), s, a.n, 2, 50, 0.0, rng)))


if __name__ == "__main__":
    main()
