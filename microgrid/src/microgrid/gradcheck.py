import numpy as np


def gradcheck(fn, inputs, eps=1e-6, tol=1e-5):
    """Compare autodiff grads of scalar fn(*inputs) with central differences.
    Returns the max relative error."""
    for t in inputs:
        t.zero_grad()
    fn(*inputs).backward()
    worst = 0.0
    for t in inputs:
        num = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = t.data[idx]
            t.data[idx] = old + eps; hi = fn(*inputs).data.item()
            t.data[idx] = old - eps; lo = fn(*inputs).data.item()
            t.data[idx] = old
            num[idx] = (hi - lo) / (2 * eps)
        err = np.abs(num - t.grad) / (np.abs(num) + np.abs(t.grad) + 1e-8)
        worst = max(worst, float(err.max()))
    return worst
