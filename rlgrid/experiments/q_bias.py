"""Does Double DQN reduce overestimation? Measure signed Q error vs exact Q* on a slippery grid."""
import numpy as np
from rlgrid import GridWorld, value_iteration
from rlgrid.dqn import dqn, q_bias

LAYOUT = ["S...", ".#..", "...#", "...G"]
rows = {}
for name, kw in [("vanilla", {}), ("double", {"double": True})]:
    res = []
    for seed in range(5):
        env = GridWorld(layout=LAYOUT, slip=0.2, seed=seed)
        net, phi, _ = dqn(env, episodes=300, seed=seed, **kw)
        res.append(q_bias(env, net, phi, value_iteration(env)))
    rows[name] = np.array(res)
print("| variant | mean signed err | RMSE | greedy agreement |\n|---|---|---|---|")
for k, v in rows.items():
    m = v.mean(0)
    print(f"| {k} | {m[0]:+.3f} | {m[1]:.3f} | {m[2]:.2f} |")
