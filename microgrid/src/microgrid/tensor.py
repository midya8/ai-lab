"""Reverse-mode autodiff over numpy arrays, with broadcasting-aware gradients."""
import numpy as np


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, _parents=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = _parents
        self._backward = _backward

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _make(self, data, parents, bw):
        return Tensor(data, parents, bw)

    def __add__(self, o):
        o = Tensor._wrap(o)
        out = Tensor(self.data + o.data, (self, o))
        def bw():
            self.grad += _unbroadcast(out.grad, self.shape)
            o.grad += _unbroadcast(out.grad, o.shape)
        out._backward = bw
        return out
    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-Tensor._wrap(o))

    def __rsub__(self, o):
        return Tensor._wrap(o) - self

    def __mul__(self, o):
        o = Tensor._wrap(o)
        out = Tensor(self.data * o.data, (self, o))
        def bw():
            self.grad += _unbroadcast(out.grad * o.data, self.shape)
            o.grad += _unbroadcast(out.grad * self.data, o.shape)
        out._backward = bw
        return out
    __rmul__ = __mul__

    def __truediv__(self, o):
        return self * Tensor._wrap(o) ** -1.0

    def __rtruediv__(self, o):
        return Tensor._wrap(o) * self ** -1.0

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Tensor(self.data ** p, (self,))
        def bw():
            self.grad += out.grad * p * self.data ** (p - 1)
        out._backward = bw
        return out

    def __matmul__(self, o):
        o = Tensor._wrap(o)
        out = Tensor(self.data @ o.data, (self, o))
        def bw():
            g = out.grad
            a, b = self.data, o.data
            if a.ndim == 1 and b.ndim == 1:
                self.grad += g * b
                o.grad += g * a
                return
            if a.ndim == 1:
                a2, g2 = a[None, :], g[None, :] if g.ndim == b.ndim - 1 else g
                self.grad += (g2 @ b.swapaxes(-1, -2)).reshape(a.shape)
                o.grad += _unbroadcast(a2.swapaxes(-1, -2) @ g2, b.shape)
                return
            if b.ndim == 1:
                self.grad += _unbroadcast(g[..., None] * b, a.shape)
                o.grad += _unbroadcast((a * g[..., None]).reshape(-1, a.shape[-1]).sum(0), b.shape)
                return
            self.grad += _unbroadcast(g @ b.swapaxes(-1, -2), a.shape)
            o.grad += _unbroadcast(a.swapaxes(-1, -2) @ g, b.shape)
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def _unary(self, f, df):
        out = Tensor(f(self.data), (self,))
        def bw():
            self.grad += out.grad * df(self.data, out.data)
        out._backward = bw
        return out

    def relu(self):
        return self._unary(lambda x: np.maximum(x, 0), lambda x, y: (x > 0).astype(float))

    def tanh(self):
        return self._unary(np.tanh, lambda x, y: 1 - y ** 2)

    def exp(self):
        return self._unary(np.exp, lambda x, y: y)

    def log(self):
        return self._unary(np.log, lambda x, y: 1 / x)

    def sigmoid(self):
        return self._unary(lambda x: 1 / (1 + np.exp(-x)), lambda x, y: y * (1 - y))

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))
        def bw():
            self.grad += out.grad.reshape(self.shape)
        out._backward = bw
        return out

    def log_softmax(self, axis=-1):
        m = self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(self.data - m).sum(axis=axis, keepdims=True)) + m
        out = Tensor(self.data - lse, (self,))
        def bw():
            sm = np.exp(out.data)
            self.grad += out.grad - sm * out.grad.sum(axis=axis, keepdims=True)
        out._backward = bw
        return out

    def backward(self):
        """Iterative post-order topological sort (no recursion limit issues)."""
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            t, done = stack.pop()
            if done:
                order.append(t)
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for p in t._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._backward is not None:
                t._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"
