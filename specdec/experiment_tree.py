"""Tokens per target call for m-ary draft trees (3 seeds). Simulated n-gram models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from specdec import NgramLM, MixtureLM, tree_generate

rng0 = np.random.default_rng(0)
toks = []
for _ in range(4000):
    toks.append(int((toks[-1] * 3 + rng0.integers(0, 3)) % 8) if toks else 0)
T = NgramLM(8, 3, 0.05).fit(toks)
U = NgramLM(8, 1, 1.0).fit(toks)
print("w_target | m=1 m=2 m=4 m=8   (tokens per target call, depth 4; draft nodes/token in brackets for m=8)")
for w in (0.2, 0.6, 0.9):
    D = MixtureLM(T, U, w)
    row = []
    for m in (1, 2, 4, 8):
        st = {}
        for s in range(3):
            tree_generate(T, D, [0], 400, 4, m, np.random.default_rng(s), st)
        row.append((1200 / st["target_calls"], st["draft_nodes"] / 1200))
    print(f"{w:8.1f} | " + " ".join(f"{r[0]:.2f}" for r in row) + f"  [{row[-1][1]:.1f}]")
