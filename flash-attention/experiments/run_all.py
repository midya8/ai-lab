"""Measured claims for flash_attention. Writes experiments/results.txt (~1 min).

1. I/O: elements moved by naive vs tiled attention, and the effect of tile size.
2. Peak memory (tracemalloc): naive N x N scores vs one tile.
3. Tile skipping: causal / sliding-window compute as a fraction of dense.
4. Numerics: float32 error vs a float64 reference, and the un-stabilised softmax failure point.
5. Split-KV decode and paged-cache accounting.
"""
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flash_attention import (PagedKVCache, flash_decode, flash_forward, naive_attention,  # noqa: E402
                      naive_traffic)

out = []


def P(s=""):
    print(s)
    out.append(s)


def rand(H, n, d, seed=0, nk=None):
    r = np.random.default_rng(seed)
    return r.normal(size=(H, n, d)), r.normal(size=(H, nk or n, d)), r.normal(size=(H, nk or n, d))


P("1. Slow-memory traffic (elements, 1 head, d=64, non-causal), naive vs tiled")
P(f"{'N':>6} {'tile':>6} {'naive':>14} {'tiled':>14} {'ratio':>7}")
d = 64
for n in (512, 1024, 2048):
    q, k, v = rand(1, n, d)
    nv = naive_traffic(n, n, d)["total"]
    for t in (16, 64, 256):
        _, _, st = flash_forward(q, k, v, block_q=t, block_k=t)
        P(f"{n:>6} {t:>6} {nv:>14,} {st.traffic:>14,} {nv / st.traffic:>7.2f}")
P("   (tiled traffic ~ N*d + (N/Br)*2*N*d: larger Br cuts K/V re-reads; N^2 score traffic is gone.)")
P("   Elements, not seconds: NumPy has no fast on-chip memory, so this is the algorithm's I/O model.")

P()
P("2. Peak Python/NumPy allocation (tracemalloc), 1 head, d=64")
P(f"{'N':>6} {'naive MB':>10} {'tiled(64) MB':>13}")
for n in (512, 1024, 2048):
    q, k, v = rand(1, n, d)
    tracemalloc.start()
    naive_attention(q, k, v)
    naive_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    tracemalloc.start()
    flash_forward(q, k, v, block_q=64, block_k=64)
    flash_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    P(f"{n:>6} {naive_peak / 1e6:>10.1f} {flash_peak / 1e6:>13.2f}")
P("   (naive also materialises probs and a mask; tiled holds q/k/v/out plus one 64x64 tile.)")

P()
P("3. Executed tile-flops as a fraction of dense (N=1024, tile 32, 1 head)")
q, k, v = rand(1, 1024, 32)
dense = flash_forward(q, k, v, block_q=32, block_k=32)[2].flops
for name, kwargs in [("causal", dict(causal=True)), ("causal, window 128", dict(causal=True, window=128)),
                     ("causal, window 32", dict(causal=True, window=32)),
                     ("bidirectional, window 128", dict(window=128))]:
    st = flash_forward(q, k, v, block_q=32, block_k=32, **kwargs)[2]
    P(f"  {name:<28} {st.flops / dense:>6.3f}  ({st.tiles_skipped}/{st.tiles_total} tiles skipped)")

P()
P("4. Numerics: max |error| vs float64 naive reference (causal, d=64, mean of 3 seeds)")
P(f"{'N':>6} {'score scale':>12} {'tiled f32':>11} {'naive f32':>11}")
for n in (256, 1024):
    for sc in (1.0, 8.0):
        e_t, e_n = [], []
        for seed in range(3):
            q, k, v = rand(1, n, 64, seed)
            q = q * sc
            ref = naive_attention(q, k, v, causal=True)[0]
            e_t.append(np.abs(flash_forward(q, k, v, causal=True, dtype=np.float32, block_q=64,
                                            block_k=64)[0] - ref).max())
            q32, k32, v32 = (x.astype(np.float32) for x in (q, k, v))
            s = (q32[0] @ k32[0].T) * np.float32(64 ** -0.5)
            s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
            p = np.exp(s - s.max(1, keepdims=True))
            e_n.append(np.abs((p / p.sum(1, keepdims=True)) @ v32[0] - ref[0]).max())
        P(f"{n:>6} {sc:>12.1f} {np.mean(e_t):>11.2e} {np.mean(e_n):>11.2e}")
P("   Both are float32 with max-subtraction, so both sit at float32 rounding; tiling does not")
P("   cost accuracy. Without the max subtraction float32 exp overflows once scores exceed ~88:")
q, k, v = rand(1, 64, 64)
for sc in (1.0, 10.0, 40.0):
    s = (q * sc)[0].astype(np.float32) @ k[0].astype(np.float32).T * np.float32(64 ** -0.5)
    with np.errstate(over="ignore"):
        bad = not np.all(np.isfinite(np.exp(s)))
    ok = np.all(np.isfinite(flash_forward(q * sc, k, v, dtype=np.float32)[0]))
    P(f"  score scale {sc:>5.1f}: max score {s.max():>7.1f}  unstabilised exp overflows: {bad!s:<5}  online softmax finite: {ok}")

P()
P("5. Split-KV decode: max |diff| vs one-pass attention, 1 query over 4096 cached tokens")
q, k, v = rand(8, 1, 64, nk=4096)
ref = naive_attention(q, k, v, causal=True)[0]
for sp in (1, 4, 16, 64):
    o = flash_decode(q, k, v, num_splits=sp, block_k=64)[0]
    P(f"  splits {sp:>3}: {np.abs(o - ref).max():.2e}")

P()
P("6. Paged cache vs contiguous reservation (block 16, max_len 512 reserved per sequence)")
r = np.random.default_rng(0)
lens = r.integers(20, 400, size=32)
c = PagedKVCache(4096, 16, 2, 8)
for n in lens:
    s = c.new_sequence()
    c.append(s, r.normal(size=(int(n), 2, 8)), r.normal(size=(int(n), 2, 8)))
u = c.utilisation()
P(f"  32 sequences, {int(lens.sum())} tokens: paged slots {u['physical_slots']}, "
  f"contiguous max_len slots {32 * 512}  ({u['physical_slots'] / (32 * 512):.1%}); "
  f"internal waste {u['internal_waste_slots']} slots")
root = c.new_sequence()
c.append(root, r.normal(size=(256, 2, 8)), r.normal(size=(256, 2, 8)))
before = c.blocks_in_use()
kids = [c.fork(root) for _ in range(8)]
for kid in kids:
    c.append(kid, r.normal(size=(8, 2, 8)), r.normal(size=(8, 2, 8)))
P(f"  8 forks of a 256-token prefix + 8 tokens each: +{c.blocks_in_use() - before} blocks "
  f"(8 full copies would be +{8 * 17}); copy-on-write events {c.copies}")

with open(os.path.join(os.path.dirname(__file__), "results.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
