"""Transformer building blocks on the microgrid autodiff engine."""
import numpy as np
from .tensor import Tensor
from .nn import Linear


def transpose(x, a, b):
    """Swap two axes."""
    out = Tensor(np.swapaxes(x.data, a, b), (x,))
    def bw():
        x.grad += np.swapaxes(out.grad, a, b)
    out._backward = bw
    return out


def take_rows(table, ids):
    """Embedding lookup: table[ids] with scatter-add backward."""
    ids = np.asarray(ids)
    out = Tensor(table.data[ids], (table,))
    def bw():
        np.add.at(table.grad, ids, out.grad)
    out._backward = bw
    return out


def softmax(x, axis=-1):
    m = x.data.max(axis=axis, keepdims=True)
    e = np.exp(x.data - m)
    out = Tensor(e / e.sum(axis=axis, keepdims=True), (x,))
    def bw():
        g, y = out.grad, out.data
        x.grad += y * (g - (g * y).sum(axis=axis, keepdims=True))
    out._backward = bw
    return out


def gelu(x):
    c = np.sqrt(2 / np.pi)
    def f(v):
        return 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v ** 3)))
    def df(v, y):
        t = np.tanh(c * (v + 0.044715 * v ** 3))
        return 0.5 * (1 + t) + 0.5 * v * (1 - t ** 2) * c * (1 + 3 * 0.044715 * v ** 2)
    return x._unary(f, df)


def masked_fill(x, mask, value):
    """Where mask is True the output is `value` and no gradient flows."""
    mask = np.broadcast_to(mask, x.shape)
    out = Tensor(np.where(mask, value, x.data), (x,))
    def bw():
        x.grad += np.where(mask, 0.0, out.grad)
    out._backward = bw
    return out


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.g, self.b, self.eps = Tensor(np.ones(d)), Tensor(np.zeros(d)), eps

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        c = x - mu
        var = (c * c).mean(axis=-1, keepdims=True)
        return c * ((var + self.eps) ** -0.5) * self.g + self.b

    def parameters(self):
        return [self.g, self.b]


class CausalSelfAttention:
    def __init__(self, d, heads, rng):
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv, self.proj = Linear(d, 3 * d, rng), Linear(d, d, rng)

    def __call__(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh)
        q, k, v = (transpose(_select(qkv, i), 1, 2) for i in range(3))
        att = (q @ transpose(k, -1, -2)) * (1.0 / np.sqrt(self.dh))
        att = masked_fill(att, np.triu(np.ones((T, T), bool), 1), -1e9)
        y = transpose(softmax(att) @ v, 1, 2).reshape(B, T, D)
        return self.proj(y)

    def parameters(self):
        return self.qkv.parameters() + self.proj.parameters()


def _select(x, i):
    """x[:, :, i] with scatter backward."""
    out = Tensor(x.data[:, :, i], (x,))
    def bw():
        x.grad[:, :, i] += out.grad
    out._backward = bw
    return out


class Block:
    def __init__(self, d, heads, rng):
        self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads, rng)
        self.fc1, self.fc2 = Linear(d, 4 * d, rng), Linear(4 * d, d, rng)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(gelu(self.fc1(self.ln2(x))))

    def parameters(self):
        return (self.ln1.parameters() + self.ln2.parameters() + self.attn.parameters()
                + self.fc1.parameters() + self.fc2.parameters())


class MiniGPT:
    def __init__(self, vocab, d=32, heads=4, layers=2, ctx=16, seed=0):
        rng = np.random.default_rng(seed)
        self.tok = Tensor(rng.normal(0, 0.1, (vocab, d)))
        self.pos = Tensor(rng.normal(0, 0.1, (ctx, d)))
        self.blocks = [Block(d, heads, rng) for _ in range(layers)]
        self.lnf = LayerNorm(d)
        self.head = Linear(d, vocab, rng)

    def __call__(self, ids):
        ids = np.asarray(ids)
        B, T = ids.shape
        x = take_rows(self.tok, ids) + take_rows(self.pos, np.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def parameters(self):
        ps = [self.tok, self.pos] + self.lnf.parameters() + self.head.parameters()
        for b in self.blocks:
            ps += b.parameters()
        return ps
