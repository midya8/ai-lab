# microgrid

A from-scratch reverse-mode autodiff engine on numpy (~200 lines): broadcasting-aware gradients, matmul (incl. batched/vector cases), reductions, activations, a numerically stable `log_softmax`, SGD+momentum, Adam, small MLP, and a finite-difference `gradcheck` used to verify every op.

```python
from microgrid import Tensor, MLP, Adam, mse_loss
m = MLP([2, 8, 1]); opt = Adam(m.parameters(), lr=0.05)
loss = mse_loss(m(Tensor(X)), y); loss.backward(); opt.step()
```

`conv2d` (stride, zero padding, bias) and `max_pool2d` are gradient-checked too; `backward()` is an iterative topological sort (a 5000-node chain works).

Transformer blocks (`attn.py`): `transpose`, `softmax`, `gelu`, `take_rows` (embedding), `masked_fill`, `LayerNorm`, `CausalSelfAttention`, `Block` (pre-LN) and `MiniGPT`, all gradient-checked; a causality test confirms later tokens never affect earlier outputs, and MiniGPT memorises a cyclic sequence (a smoke test of learning, not a language-modelling result).

Char-level LM training (`lm.py`, `experiments/train_corpus.py`): trains MiniGPT (d=32, 4 heads, 2 layers, ctx=32) on tiny-transformer's corpus with Adam. In a 400-step run (~10 s, CPU) val loss went 3.76 -> 0.19 (uniform baseline 3.61). Honest reading: the corpus is 2.6 KB and repetitive, so the 'held-out' tail overlaps the training text and the model largely memorises it (generation reproduces the passage); this shows the hand-written autodiff trains a transformer end to end, not generalisation. Parity with tiny-transformer (`parity.py`, `tests/test_parity.py`): weights of a tinygpt.GPT (float64, randomised so biases/LN are non-trivial) are copied into MiniGPT; logits, loss and all 30 parameter gradients agree (logits/loss to <1e-9, grads to <1e-8) between hand-written backprop and tape autodiff. Covers the plain config only (learned positions, MHA, untied head, no dropout).

Test: `PYTHONPATH=src python -m pytest -q` (26 tests incl. LM training/generation smoke tests; 24 earlier: the original 16: gradient checks for each op incl. conv/pool, conv vs naive loop, XOR via MLP, a 2-blob classifier, a tiny CNN on synthetic row-vs-column stripes).

Limits: float64 CPU only, conv is einsum/strided-view based (fine for small images, not fast), no graph freeing. Educational, not a framework.

## Optimiser extras

`AdamW` (decoupled weight decay), `clip_grad_norm`, and `cosine_warmup` are in `microgrid.optim`; `lm.train` accepts `weight_decay`, `clip`, `warmup`, `schedule`. Tests check the exact decay step, AdamW==Adam at wd=0, clip norm, and schedule shape; no quality claim is made for them on the toy corpus.

## Leak-free held-out run

`experiments/train_heldout.py` trains on Python stdlib source (even-indexed files, 60k chars) and validates on disjoint files (odd-indexed, 12k chars) via `make_data_split` (vocab from train only). One 600-step run (d=48, 2 layers, ctx=48, clip+warmup+cosine, ~30 s CPU, single seed): held-out loss 5.35 -> 2.47 (uniform 4.53), train ~2.03. Samples are code-flavoured gibberish (indentation, quotes, common words); a tiny model on 60k chars, so this shows real generalisation to unseen files but not quality. Train loss is a single noisy batch.

## KV-cache inference (`kvcache.py`)

Pure-numpy forward with per-layer key/value caches (`prefill`, `step`, `generate_cached`). Tests check it matches the autodiff forward to 1e-9 at every position, chunked prefill equals single prefill, and `generate_cached` reproduces `lm.generate` token for token, including after the window slides past ctx (absolute positions force a re-prefill there). `experiments/bench_kvcache.py` (d=64, 4 layers, CPU): ctx=64 12x faster (0.33s -> 0.03s), ctx=256 33x (4.14s -> 0.12s). Caveat: the baseline also pays autodiff tape overhead that the cached path skips, so the speedup mixes both effects; it is not caching alone. Also: `QuantKVCache` (int8 values + one float32 scale per head/token vector; ~3.6x smaller than float64 at dh=16, outputs within 0.05 logits of exact in tests, so generation is no longer guaranteed token-identical to `lm.generate`), and `top_k` / `top_p` sampling in `generate_cached` (defaults leave the sampling stream unchanged). Tests use small random models; no quality (perplexity) evaluation of the int8 cache was run. 52 tests.

## Multi-seed + weight-decay ablation (`experiments/seeds_heldout.py`)

Same leak-free split, 400 steps, 3 seeds (seed varies init and batch sampling). Held-out loss: wd=0.0 -> 2.568 +- 0.018 (runs 2.576/2.544/2.585); wd=0.1 -> 2.558 +- 0.017 (2.566/2.534/2.574). Uniform baseline 4.533. wd=0.1 is lower by ~0.01 in every paired seed, but that is well inside the seed-to-seed spread and 3 seeds is few, so this is a null result: weight decay has no demonstrable effect at this scale. The single-run 2.47 above (600 steps) was a more favourable draw/longer run, not comparable.

## int8 KV cache perplexity (`experiments/kv_perplexity.py`)

`cached_nll` teacher-forces a sequence one token at a time through a cache (windows of ctx), so every attention read uses what the cache stored; with the exact cache it matches the autodiff loss to 1e-9 (tested). On 3000 held-out stdlib characters, 400-step models, 3 seeds: exact vs int8 NLL differ by -0.00006 / +0.00016 / +0.00002 nats (perplexity ~15.65-15.70 either way). Reading: int8 per-vector quantisation of K/V costs nothing measurable here, but these are tiny models (dh=12, ctx 48) with short caches, so this does not predict behaviour at scale. 53 tests.
