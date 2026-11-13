"""Episodes needed to first reach the optimal greedy return, Dyna-Q vs prioritized sweeping vs Q-learning.
Default grid, slip 0, 10 seeds, planning=10 for both planners, alpha=0.5 eps=0.1 (library defaults, fixed
beforehand). Q is snapshotted by re-running with growing episode counts (same seed => same prefix)."""
import numpy as np
from rlgrid import *

env0 = GridWorld()
opt = evaluate(env0, greedy_policy(value_iteration(env0)), 1)[0]

def first_optimal(fn, seed, cap=120, **kw):
    for n in range(1, cap + 1):
        res = fn(GridWorld(seed=seed), episodes=n, seed=seed, **kw)
        q = res[0]
        if abs(evaluate(GridWorld(seed=seed), greedy_policy(q), 1)[0] - opt) < 1e-9:
            return n, (res[2] if len(res) > 2 else None)
    return cap + 1, None

for name, fn in [("q_learning", q_learning), ("dyna_q", dyna_q), ("prioritized_sweeping", prioritized_sweeping)]:
    out = [first_optimal(fn, s) for s in range(10)]
    eps = [o[0] for o in out]
    extra = [o[1] for o in out if o[1] is not None]
    print(f"{name:22s} episodes to optimal: mean {np.mean(eps):5.1f} median {np.median(eps):4.1f}"
          + (f"  planning backups {np.mean(extra):.0f}" if extra else ""))
