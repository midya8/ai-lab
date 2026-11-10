"""Wall-clock: cached vs full-window generation (fits-in-ctx regime). CPU, numpy."""
import time
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import generate
from microgrid.kvcache import generate_cached

for ctx in (64, 256):
    m = MiniGPT(40, d=64, heads=4, layers=4, ctx=ctx, seed=0)
    chars = [chr(97 + i % 26) for i in range(40)]
    n = ctx - 4
    t = time.perf_counter(); a = generate(m, chars, [1, 2, 3, 4], n, ctx, seed=1); t_full = time.perf_counter() - t
    t = time.perf_counter(); b = generate_cached(m, chars, [1, 2, 3, 4], n, ctx, seed=1); t_kv = time.perf_counter() - t
    print(f"ctx={ctx} n={n} identical={a == b} full={t_full:.2f}s cached={t_kv:.2f}s speedup={t_full / t_kv:.1f}x")
