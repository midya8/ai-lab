"""Acceptance of PromptLookupLM (n=1,2,3) vs BackoffLookupLM on a repetitive-phrase target.
Target: order-3 NgramLM fit on text made of repeated random phrases (synthetic)."""
import numpy as np
from specdec import *

V = 12
r = np.random.default_rng(0)
phrases = [list(r.integers(0, V, 8)) for _ in range(6)]
text = []
for _ in range(300):
    text += phrases[r.integers(0, 6)]
t = NgramLM(V, 3, alpha=0.05).fit(text)
drafts = {f"lookup n={n}": PromptLookupLM(V, n=n) for n in (1, 2, 3)}
drafts["backoff 3..1"] = BackoffLookupLM(V, n_max=3, n_min=1)
for name, d in drafts.items():
    acc = prop = 0
    for s in range(20):
        st = {}
        rng = np.random.default_rng(s)
        speculative_generate(t, d, phrases[0] * 2, 200, 4, rng, stats=st)
        acc += st["accepted"]; prop += st["proposed"]
    print(f"{name:14s} acceptance {acc / prop:.3f}")
