# flashlab

The attention-kernel ideas behind FlashAttention, FlashDecoding and PagedAttention, written from scratch in NumPy and **checked against a naive reference to ~1e-12**. Nothing here is fast (it is NumPy on a CPU); what it demonstrates is that the algorithms are *exact*, and it measures the I/O, memory and compute they save with counters rather than asserting it.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q          # 100 tests (~1 s)
python experiments/run_all.py               # regenerates experiments/results.txt (~1 min)
python experiments/kv_quant.py              # regenerates experiments/kv_quant_results.txt (seconds)
```

```python
import numpy as np
from flashlab import flash_forward, flash_backward, flash_decode, PagedKVCache, alibi_slopes

q = np.random.randn(8, 512, 64); k = np.random.randn(2, 512, 64); v = np.random.randn(2, 512, 64)  # GQA: 8 q heads, 2 kv heads
out, lse, stats = flash_forward(q, k, v, causal=True, window=128, alibi_slopes=alibi_slopes(8),
                                block_q=64, block_k=64)
print(stats.skipped_fraction, stats.traffic)   # tiles never computed, elements moved
dq, dk, dv = flash_backward(q, k, v, out, lse, np.ones_like(out), causal=True, window=128,
                            alibi_slopes=alibi_slopes(8))   # recomputes P from lse, tile by tile
```

## What is in it

| module | contents |
|---|---|
| `reference.py` | naive attention (full score matrix), its analytic backward, and the closed-form traffic of the 3-kernel implementation. Supports causal, sliding window, ALiBi, grouped/multi-query attention |
| `masks.py` | the mask rules and `tile_status`: classifies a (query-tile x key-tile) block as skip / full / partial in O(1) - tested against brute force on every combination |
| `flash.py` | `attend_partial` (online softmax over key tiles, running max and denominator), exact `merge_partials`, `flash_forward`, FlashAttention-2 style `flash_backward` (stores only the log-sum-exp, recomputes P per tile), and `Stats` counters (tiles, elements read/written, flops) |
| `decode.py` | `flash_decode`: split-KV decoding for one query over a long cache, partial results merged exactly |
| `paged.py` | `PagedKVCache`: block pool, per-sequence block tables, fork with prefix sharing, copy-on-write, attention that runs block-by-block without gathering the cache. `QuantPagedKVCache` is the same cache storing bit-packed low-bit K and V (independent `k_bits`/`v_bits`) and dequantising one block at a time |
| `quant.py` | symmetric grouped quantisation (2/4/8 bit) along the head dimension, float16 scales rounded *up* so nothing clips, real bit-packing, and `bytes_per_vector` accounting |

## What the tests establish

* Tiled forward equals naive for 6 mask configurations x 4 tile shapes (including 1x1 and tiles that do not divide N) to 1e-12; `lse` too.
* Tiled backward equals the analytic backward to 1e-10 for all configurations, and the analytic backward is itself checked against central finite differences (so the chain is FD -> naive -> tiled).
* Rows with **no visible key** give zeros and `lse = -inf` in both implementations and produce zero, finite gradients (the classic `-inf - -inf = NaN` bug).
* Merging partial states is independent of order and grouping, and survives a split in which every score is masked.
* GQA equals multi-head attention with the KV heads repeated.
* Counters are checked against closed forms (unmasked traffic exactly `H*(Nq*d + ceil(Nq/Br)*2*Nk*d)` reads; causal skips exactly the strictly-upper tiles; window attention flops scale linearly in N).
* Quantisation: every element's error is at most half its group scale (all bit widths x group sizes); pack/unpack round-trips every code exhaustively; error falls with bits and with smaller groups when a channel is an outlier; a token's codes do not depend on other tokens, so appends never disturb old tokens; attention over the quantised blocks equals exact attention over the dequantised cache to 1e-12 (the online softmax adds no error of its own); fork/copy-on-write/out-of-blocks behave as in the full-precision cache; byte counts equal the real array sizes.
* Paged cache: gather round-trips across block boundaries; paged attention equals contiguous attention; forks share blocks until written; copy-on-write isolates parent and child; freeing a shared block only releases it at refcount 0; running out of blocks raises without half-applying the append.

## Measured results (`experiments/results.txt`)

**I/O (elements moved, 1 head, d=64).** Tiling removes the N x N score traffic but re-reads K and V once per query tile, so the tile size matters: at tile 64 traffic is exactly 2.00x lower than the naive 3-kernel pipeline at every N tested; at tile 256 it is 6.0x (N=512) to 7.3x (N=2048) lower; **at tile 16 it is about 2x *worse* than naive**, because K/V are re-read N/16 times. The saving is real only when tiles are big enough - on a GPU, that is what the on-chip SRAM size limits.

**Peak allocation (tracemalloc).** Naive at N=2048: 173 MB (scores + probabilities + mask); tiled with 64x64 tiles: 3.2 MB, growing linearly in N rather than quadratically.

**Compute skipped by masks (N=1024, tile 32).** Causal runs 51.6% of the dense tile-flops (the diagonal tiles are the overhead above 50%); causal with a 128-token window 14.6%; window 32 6.2%; bidirectional window 128 26.2%.

**Numerics.** float32 tiled and float32 naive (both with max-subtraction) are indistinguishable from each other: max error vs float64 is ~5e-7 at unit-scale scores and ~1e-5 at 8x-scaled queries, with no accuracy cost from tiling. The online softmax is finite where an un-stabilised float32 `exp` overflows (scores of ~147 at scale 40).

**Split-KV decode.** 1, 4, 16 and 64 splits over 4096 cached tokens all agree with one-pass attention to ~3e-16.

**Paged cache.** 32 sequences of 20-400 tokens occupy 44.5% of the slots that reserving max_len=512 each would (7,296 vs 16,384; 264 slots of internal waste, under one block per sequence). Eight forks of a 256-token prefix each appending 8 tokens cost +8 blocks instead of +136 for full copies.

**KV-cache quantisation (`experiments/kv_quant_results.txt`, synthetic data).** Keys with 4 outlier channels (x8), 4 kv heads, d=64, 1024 cached tokens, 5 seeds; fp16 K+V = 256 bytes per token-head. Relative attention-output error / compression (group 16 scales):

| K / V bits | bytes | compression | diffuse queries | peaky queries (x3) |
|---|---|---|---|---|
| 8 / 8 | 144 | 1.78x | 0.021 | 0.034 |
| 8 / 4 | 112 | 2.29x | 0.087 | 0.091 |
| 4 / 8 | 112 | 2.29x | 0.33 | 0.53 |
| 4 / 4 | 80 | 3.20x | 0.34 | 0.54 |
| 2 / 2 | 48 | 5.33x | 0.98 | 1.14 |

What it shows: **keys are far more sensitive than values** (4-bit K alone: 0.33 diffuse / 0.53 peaky; 4-bit V alone: 0.085 in both), because K error moves the softmax logits and peaky queries amplify it; the lse error at K=4 is 0.7 (diffuse) to 3.6 (peaky) versus 0.04-0.4 at K=8. 2-bit is unusable here (error ~ 1, i.e. the output is noise) and its scale overhead already eats a third of the bytes. Smaller scale groups (16 vs a per-token 64) cut error 1.3-1.7x for ~12 extra bytes; per-token scales make the outlier channels set everyone's resolution. An offline KIVI-style layout (scale per *channel* over 16-token blocks, same 8 bytes of scales per token-head) is 2.6x better at 4-bit keys (0.12 vs 0.33 diffuse) but *worse* at 2 bits (0.95 vs 0.76), and it is **not** implemented in the cache - it needs a full-precision residual buffer for the partly filled block, since scales cannot be known until the block is full.

## Honest limits

* The quantisation study is synthetic: the outlier-channel structure is an assumption modelled on what KV-quantisation papers report, not measured on a real model, and "relative output error" is not perplexity or task accuracy. The ranking (K needs more bits than V; scale granularity matters) is a property of the mechanism on this data, not a claim about any particular LLM. Under diffuse attention the exact output is a small average of many values, so relative error is a demanding metric.
* Quantised blocks are dequantised in float64 NumPy; nothing here is a fused int8 kernel, so there is no speed claim, only bytes and error.
* No radix-tree prefix cache, no batch dimension yet.

* Wall-clock is not the point: one unrepeated timing at N=2048, d=64, 1 head gave naive 0.07 s, tiled(64) 0.09 s, tiled(256) 0.05 s - a wash on a CPU, and not a benchmark. The I/O, memory and flop numbers are model/counter measurements of the algorithm, not hardware results. Real FlashAttention wins come from fused GPU kernels this repo does not contain.
* "Slow memory traffic" counts elements read/written under the usual FlashAttention I/O model (Q, K, V tiles in; O and lse out); it ignores caches, dtype width and L2 effects.
* Single sequence, no batch dimension, no dropout, no head dimension > what NumPy matmul handles comfortably.
