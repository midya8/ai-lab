"""Fixed k vs AdaptiveK. Modeled cost per token = (target_calls + c*proposed)/n, c=0.1 (simulated)."""
import numpy as np
from specdec import *

V, C, N = 12, 0.1, 4000
rng = np.random.default_rng(0)
T = rng.dirichlet(np.ones(V) * 0.15, size=(V, V))
toks = [0, 1]
for _ in range(20000):
    toks.append(int(rng.choice(V, p=T[toks[-2], toks[-1]])))
target = NgramLM(V, 3, 0.05).fit(toks)
weak = NgramLM(V, 1, 1.0).fit(toks)


def cost(st):
    return (st["target_calls"] + C * st["proposed"]) / N


print("w_target | cost/token (lower is better; autoregressive = 1.000)")
print("         |  k=2   k=4   k=8  k=16 adaptive (mean k)")
for w in (0.0, 0.5, 0.8, 0.95):
    draft = MixtureLM(target, weak, w)
    row = []
    for k in (2, 4, 8, 16):
        cs = []
        for s in range(3):
            st = {}
            speculative_generate(target, draft, [0, 1], N, k, np.random.default_rng(s), st)
            cs.append(cost(st))
        row.append(np.mean(cs))
    cs, ks = [], []
    for s in range(3):
        st = {}
        ctl = AdaptiveK(C)
        speculative_generate(target, draft, [0, 1], N, 0, np.random.default_rng(s), st, controller=ctl)
        cs.append(cost(st)); ks.append(st["proposed"] / st["target_calls"])
    print(f"{w:8.2f} | " + " ".join(f"{x:5.3f}" for x in row) + f"  {np.mean(cs):5.3f} ({np.mean(ks):.1f})")
