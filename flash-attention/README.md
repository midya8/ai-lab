# flash-attention

The ideas behind FlashAttention, FlashDecoding and PagedAttention, written in NumPy so I could see exactly why they work. Nothing here is fast. The point is that the tiled versions give the same answer as naive attention (to ~1e-12) and you can count the memory traffic they save.

What's in it: online-softmax tiling with causal, sliding-window, ALiBi and GQA masks (fully masked tiles are skipped), a FlashAttention-2 style backward that only keeps the log-sum-exp, split-KV decoding, and a paged KV cache with forking and copy-on-write that can also store int8/int4/int2 keys and values.

```bash
cd flash-attention
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q     # 100 tests, ~1 s
python experiments/run_all.py          # regenerates experiments/results.txt
python experiments/kv_quant.py         # KV quantisation study
```

```python
import numpy as np
from flash_attention import flash_forward, flash_backward, alibi_slopes

q = np.random.randn(8, 512, 64); k = np.random.randn(2, 512, 64); v = np.random.randn(2, 512, 64)
out, lse, stats = flash_forward(q, k, v, causal=True, window=128,
                                alibi_slopes=alibi_slopes(8), block_q=64, block_k=64)
print(stats.skipped_fraction, stats.traffic)
```

## Things I measured

- Traffic: with 64-wide tiles, tiled attention moves exactly 2x fewer elements than the naive three-kernel version. With 256-wide tiles it's 6-7x. With 16-wide tiles it's about 2x *worse*, because K and V get re-read for every query tile. Tile size really matters.
- Peak memory at N=2048: 173 MB naive vs 3.2 MB tiled.
- Causal masking at N=1024 runs 51.6% of the dense flops, causal plus a 128 window runs 14.6%.
- Paged cache: 32 sequences of 20-400 tokens use 44.5% of the slots a fixed max_len=512 reservation would. Eight forks of a shared 256-token prefix cost 8 new blocks instead of 136.
- KV quantisation on synthetic data with outlier channels: keys are much more sensitive than values. 4-bit V alone gives ~0.085 relative output error, 4-bit K alone gives 0.33 (0.53 with peaky queries). 2-bit is basically noise.

## Caveats

Wall-clock on a CPU is a wash and isn't the point. The I/O numbers come from counters under the usual FlashAttention I/O model, not hardware. The quantisation study uses synthetic outlier channels, not a real model's cache. No batch dimension, no radix prefix cache yet.
