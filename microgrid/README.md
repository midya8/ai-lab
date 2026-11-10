# microgrid

A from-scratch reverse-mode autodiff engine on numpy (~200 lines): broadcasting-aware gradients, matmul (incl. batched/vector cases), reductions, activations, a numerically stable `log_softmax`, SGD+momentum, Adam, small MLP, and a finite-difference `gradcheck` used to verify every op.

```python
from microgrid import Tensor, MLP, Adam, mse_loss
m = MLP([2, 8, 1]); opt = Adam(m.parameters(), lr=0.05)
loss = mse_loss(m(Tensor(X)), y); loss.backward(); opt.step()
```

`conv2d` (stride, zero padding, bias) and `max_pool2d` are gradient-checked too; `backward()` is an iterative topological sort (a 5000-node chain works).

Transformer blocks (`attn.py`): `transpose`, `softmax`, `gelu`, `take_rows` (embedding), `masked_fill`, `LayerNorm`, `CausalSelfAttention`, `Block` (pre-LN) and `MiniGPT`, all gradient-checked; a causality test confirms later tokens never affect earlier outputs, and MiniGPT memorises a cyclic sequence (a smoke test of learning, not a language-modelling result).

Char-level LM training (`lm.py`, `experiments/train_corpus.py`): trains MiniGPT (d=32, 4 heads, 2 layers, ctx=32) on tiny-transformer's corpus with Adam. In a 400-step run (~10 s, CPU) val loss went 3.76 -> 0.19 (uniform baseline 3.61). Honest reading: the corpus is 2.6 KB and repetitive, so the 'held-out' tail overlaps the training text and the model largely memorises it (generation reproduces the passage); this shows the hand-written autodiff trains a transformer end to end, not generalisation. No numerical-parity comparison with tiny-transformer yet (different init/architecture details).

Test: `PYTHONPATH=src python -m pytest -q` (26 tests incl. LM training/generation smoke tests; 24 earlier: the original 16: gradient checks for each op incl. conv/pool, conv vs naive loop, XOR via MLP, a 2-blob classifier, a tiny CNN on synthetic row-vs-column stripes).

Limits: float64 CPU only, conv is einsum/strided-view based (fine for small images, not fast), no graph freeing. Educational, not a framework.
