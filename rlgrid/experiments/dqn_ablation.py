"""DQN ablation (one-hot state features; coordinate features failed, see README): replay and target network on/off, default grid, 5 seeds, 300 episodes.
Hyperparameters fixed a priori (not tuned). Metric: greedy success rate and mean return of last 50 training episodes."""
import numpy as np
from rlgrid import GridWorld, evaluate
from rlgrid.dqn import dqn, dqn_policy

for name, kw in [("full", {}), ("no target", dict(use_target=False)), ("no replay", dict(use_replay=False)),
                 ("neither", dict(use_target=False, use_replay=False))]:
    wins, last = [], []
    for s in range(5):
        env = GridWorld(seed=s)
        net, phi, rets = dqn(env, episodes=300, seed=s, **kw)
        wins.append(evaluate(env, dqn_policy(net, phi), episodes=1)[1]); last.append(rets[-50:].mean())
    print(f"{name:10s} greedy success {np.mean(wins):.2f}  last-50 return {np.mean(last):+.3f}")
