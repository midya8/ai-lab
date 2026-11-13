"""Episodes-to-solve and final quality, 5 seeds, slip=0.1 (hyperparameters fixed a priori)."""
import numpy as np
from rlgrid import *

for name, fn in [("q_learning", q_learning), ("sarsa", sarsa), ("dyna_q(10)", dyna_q)]:
    early, wins = [], []
    for seed in range(5):
        env = GridWorld(slip=0.1, seed=seed)
        q, rets = fn(env, episodes=300, seed=seed)
        early.append(rets[:50].mean())
        wins.append(evaluate(env, greedy_policy(q), 200)[1])
    print(f"{name:12s} mean return first 50 eps {np.mean(early):+.2f}  greedy success {np.mean(wins):.2f}")
env = GridWorld()
print(env.render(greedy_policy(value_iteration(env))))
