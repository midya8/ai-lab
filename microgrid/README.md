# microgrid

A from-scratch reverse-mode autodiff engine on numpy (~200 lines): broadcasting-aware gradients, matmul (incl. batched/vector cases), reductions, activations, a numerically stable `log_softmax`, SGD+momentum, Adam, small MLP, and a finite-difference `gradcheck` used to verify every op.

```python
from microgrid import Tensor, MLP, Adam, mse_loss
m = MLP([2, 8, 1]); opt = Adam(m.parameters(), lr=0.05)
loss = mse_loss(m(Tensor(X)), y); loss.backward(); opt.step()
```

`conv2d` (stride, zero padding, bias) and `max_pool2d` are gradient-checked too; `backward()` is an iterative topological sort (a 5000-node chain works).

Test: `PYTHONPATH=src python -m pytest -q` (16 tests: gradient checks for each op incl. conv/pool, conv vs naive loop, XOR via MLP, a 2-blob classifier, a tiny CNN on synthetic row-vs-column stripes).

Limits: float64 CPU only, conv is einsum/strided-view based (fine for small images, not fast), no graph freeing. Educational, not a framework.
