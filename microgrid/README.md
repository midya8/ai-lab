# microgrid

A small reverse-mode autodiff engine on NumPy (~200 lines for the core) that I kept extending until it could train a transformer: broadcasting-aware gradients, matmul, conv2d and max-pool, LayerNorm, causal attention, a `MiniGPT`, SGD/Adam/AdamW, and a finite-difference `gradcheck` that every op is tested against.

```python
from microgrid import Tensor, MLP, Adam, mse_loss
m = MLP([2, 8, 1]); opt = Adam(m.parameters(), lr=0.05)
loss = mse_loss(m(Tensor(X)), y); loss.backward(); opt.step()
```

```bash
cd microgrid
PYTHONPATH=src python -m pytest -q          # 72 tests
python experiments/train_heldout.py         # char LM on stdlib source, ~30 s
```

## Things I tried with it

- **Parity with tiny-transformer.** Copying weights from the hand-written-backprop GPT in [`tiny-transformer`](../tiny-transformer) gives the same logits, loss and all 30 gradients to <1e-8.
- **A real held-out run.** Trained on 60k chars of Python stdlib source and validated on different files, held-out loss goes 5.35 to 2.47 (uniform is 4.53). The samples are code-flavoured gibberish, which is about right for a model this size.
- **KV cache.** Matches the full forward to 1e-9. 12x faster at ctx 64 and 33x at 256, though some of that is just skipping the autodiff tape. An int8 KV cache cost nothing measurable in perplexity.
- **Weight quantization.** 8-bit is free. 4-bit per-channel costs +0.022 nats, and per-tensor costs about 3x more. GPTQ-style error compensation roughly halves the 3-bit loss (+0.181 to +0.096), and the sequential version gets it to +0.080. 2-bit stays unusable whatever I do.
- **Weight decay** had no measurable effect across 3 seeds at this scale.

Float64 on CPU, no graph freeing, conv via strided views. It's for learning, not a framework. The quantization is simulated (real accuracy, no packed kernels).
