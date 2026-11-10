"""Prompt-lookup draft vs a unigram draft on repetitive vs random targets (3 seeds)."""
import numpy as np
from specdec import *
from specdec.lookup import PromptLookupLM

V, K, N = 8, 4, 200
for name in ("periodic", "random"):
    for dname in ("lookup", "unigram"):
        accs, tpc = [], []
        for s in range(3):
            r = np.random.default_rng(s)
            toks = [i % V for i in range(1500)] if name == "periodic" else list(r.integers(0, V, 1500))
            t = NgramLM(V, 2, alpha=0.05).fit(toks)
            d = PromptLookupLM(V, 2) if dname == "lookup" else NgramLM(V, 0, alpha=1.0).fit(toks)
            st = {}
            speculative_generate(t, d, toks[:6], N, K, np.random.default_rng(100 + s), st)
            accs.append(st["accepted"] / st["proposed"]); tpc.append(N / st["target_calls"])
        print(f"{name:9s} {dname:8s} alpha={np.mean(accs):.3f} tokens/target_call={np.mean(tpc):.2f}")
