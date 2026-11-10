# microgrid

A from-scratch reverse-mode autodiff engine on numpy (~200 lines): broadcasting-aware gradients, matmul (incl. batched/vector cases), reductions, activations, a numerically stable `log_softmax`, SGD+momentum, Adam, small MLP, and a finite-difference `gradcheck` used to verify every op.

```python
from microgrid import Tensor, MLP, Adam, mse_loss
m = MLP([2, 8, 1]); opt = Adam(m.parameters(), lr=0.05)
loss = mse_loss(m(Tensor(X)), y); loss.backward(); opt.step()
```

Test: `PYTHONPATH=src python -m pytest -q` (10 tests: gradient checks for each op, XOR learned by MLP, a 2-blob classifier).

Limits: float64 CPU only, no conv, `backward()` is recursive topological sort (very deep graphs raise the recursion limit), no graph freeing. Educational, not a framework.
