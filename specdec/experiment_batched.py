"""Wall-clock: batched vs per-position target verification (untrained nets, timing only)."""
import time
import numpy as np
from specdec.neural import MiniGPTLM
from specdec import speculative_generate, autoregressive_generate
from microgrid import MiniGPT

V, CTX = 64, 32
tgt = MiniGPTLM(MiniGPT(V, d=48, heads=4, layers=2, ctx=CTX, seed=0), CTX)
dr = MiniGPTLM(MiniGPT(V, d=16, heads=2, layers=1, ctx=CTX, seed=1), CTX)
class NoBatch:
    def __init__(s, m): s.m = m
    def probs(s, h): return s.m.probs(h)
prompt = list(range(CTX))
def t(f):
    t0 = time.perf_counter(); f(); return time.perf_counter() - t0
for k in (2, 4):
    a = t(lambda: speculative_generate(NoBatch(tgt), dr, prompt, 40, k, np.random.default_rng(0)))
    tgt.forward_calls = 0
    b = t(lambda: speculative_generate(tgt, dr, prompt, 40, k, np.random.default_rng(0)))
    print(f"k={k} per-position {a:.2f}s  batched {b:.2f}s  target forward passes {tgt.forward_calls}")
print(f"autoregressive {t(lambda: autoregressive_generate(tgt, prompt, 40, np.random.default_rng(0))):.2f}s")
