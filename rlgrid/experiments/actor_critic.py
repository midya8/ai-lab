"""Actor-critic feature comparison on the default grid (slip 0). 10 seeds, 300 episodes.
Step sizes (0.1/0.1) fixed a priori, NOT tuned per feature set. Metrics: success of the final
greedy policy, and mean training return over all episodes. Also compares to REINFORCE+baseline."""
import numpy as np
from rlgrid import *

def run(fn, **kw):
    succ, mean = [], []
    for s in range(10):
        env = GridWorld(seed=s)
        out = fn(env, episodes=300, seed=s, **kw)
        if fn is actor_critic:
            theta, _, rets, phi = out
            pol = ac_policy(theta, phi)
        else:
            theta, rets = out
            pol = greedy_policy(theta)
        succ.append(evaluate(env, pol, 1)[1]); mean.append(rets.mean())
    return np.mean(succ), np.mean(mean)

rows = [("actor-critic onehot", actor_critic, dict(features="onehot")),
        ("actor-critic coords", actor_critic, dict(features="coords")),
        ("actor-critic rbf s=0.7", actor_critic, dict(features="rbf", rbf_sigma=0.7)),
        ("actor-critic rbf s=1.0", actor_critic, dict(features="rbf", rbf_sigma=1.0)),
        ("REINFORCE+baseline (tabular)", reinforce, {})]
for name, fn, kw in rows:
    s, m = run(fn, **kw)
    print(f"| {name} | {s:.1f} | {m:+.3f} |")
