"""Cliff walking: SARSA (on-policy) vs Q-learning (off-policy) online safety gap.
Pits along the bottom edge are terminal (-1). Fixed in advance: eps=0.1, alpha=0.5, 10 seeds."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from rlgrid import GridWorld, q_learning, sarsa, greedy_policy
from rlgrid.env import CLIFF_MAP


def run(episodes=500, seeds=10, eps=0.1):
    out = {}
    for name, fn in [("Q-learning", q_learning), ("SARSA", sarsa)]:
        online, falls, path = [], [], []
        for sd in range(seeds):
            env = GridWorld(CLIFF_MAP, seed=sd)
            q, rets = fn(env, episodes=episodes, eps=eps, seed=sd, gamma=0.99)
            tail = rets[-100:]
            online.append(tail.mean()); falls.append(float((tail < -0.5).mean()))
            s = env.reset(); n = 0
            while n < 100:
                s, _, d = env.step(int(greedy_policy(q)[s])); n += 1
                if d: break
            path.append(n if env.pos in env.goals else np.nan)
        out[name] = (np.mean(online), np.mean(falls), np.nanmean(path))
    return out


if __name__ == "__main__":
    print("| agent | online return (last 100 eps) | fell in pit | greedy path length |\n|---|---|---|---|")
    for k, (o, f, p) in run().items():
        print(f"| {k} | {o:+.3f} | {f:.1%} | {p:.0f} |")
