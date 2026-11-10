"""Acceptance rate and modeled speedup vs draft quality and k (synthetic, simulated cost)."""
import numpy as np
from specdec import *

V = 12
rng = np.random.default_rng(0)
# structured source: second-order Markov chain
T = rng.dirichlet(np.ones(V) * 0.15, size=(V, V))
toks = [0, 1]
for _ in range(20000):
    toks.append(int(rng.choice(V, p=T[toks[-2], toks[-1]])))
target = NgramLM(V, 3, 0.05).fit(toks)
weak = NgramLM(V, 1, 1.0).fit(toks)
print("w_target  k  alpha_emp  tok/target_call  modeled_speedup(c=0.1)")
for w in (0.0, 0.5, 0.8, 0.95):
    draft = MixtureLM(target, weak, w)
    for k in (2, 4, 8):
        st = {}
        speculative_generate(target, draft, [0, 1], 3000, k, np.random.default_rng(1), st)
        a = st["accepted"] / st["proposed"]
        print(f"{w:7.2f} {k:3d} {a:9.3f} {3000 / st['target_calls']:15.2f} {expected_speedup(a, k, 0.1):14.2f}")
