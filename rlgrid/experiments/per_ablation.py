"""Uniform vs prioritized replay (alpha 0.6, beta 0.4, fixed a priori), default grid, 5 seeds, 300 episodes."""
import numpy as np
from rlgrid import GridWorld, evaluate
from rlgrid.dqn import dqn, dqn_policy

for name, kw in [("uniform", {}), ("prioritized", dict(prioritized=True))]:
    wins, last = [], []
    for s in range(5):
        env = GridWorld(seed=s)
        net, phi, rets = dqn(env, episodes=300, seed=s, **kw)
        wins.append(evaluate(env, dqn_policy(net, phi), episodes=1)[1]); last.append(rets[-50:].mean())
    print(f"{name:12s} greedy success {np.mean(wins):.2f}  last-50 return {np.mean(last):+.3f}")
