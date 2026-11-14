"""Exact-regret sample efficiency of tabular learners (slip 0.1, 5 seeds)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlgrid import GridWorld, q_learning, sarsa, dyna_q, expected_sarsa, regret_curve

env = GridWorld(slip=0.1)
cps = (10, 25, 50, 100, 200, 400)
algos = {"q_learning": q_learning, "sarsa": sarsa, "expected_sarsa": expected_sarsa, "dyna_q(10)": dyna_q}
print("| algo | " + " | ".join(str(c) for c in cps) + " |")
print("|---|" + "---|" * len(cps))
for name, fn in algos.items():
    c = regret_curve(env, fn, cps, seeds=range(5))
    print(f"| {name} | " + " | ".join(f"{c[k][0]:.3f}±{c[k][1]:.3f}" for k in cps) + " |")
