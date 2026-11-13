"""Learning speed vs lambda / n on the default grid (slip=0.1). 8 seeds, 60 episodes.
Hyperparameters (alpha=0.3, eps=0.1) fixed a priori; metric = mean return over the 60 episodes."""
import numpy as np
from rlgrid import *

def score(fn, **kw):
    return np.mean([fn(GridWorld(slip=0.1, seed=s), episodes=60, seed=s, **kw)[1].mean() for s in range(8)])

print("SARSA(lambda)  lam: " + "  ".join(f"{l}:{score(sarsa_lambda, lam=l):+.3f}" for l in (0, .3, .6, .8, .9)))
print("Watkins Q(lam) lam: " + "  ".join(f"{l}:{score(sarsa_lambda, lam=l, watkins=True):+.3f}" for l in (0, .3, .6, .8, .9)))
print("n-step SARSA     n: " + "  ".join(f"{n}:{score(n_step_sarsa, n=n):+.3f}" for n in (1, 2, 4, 8)))
