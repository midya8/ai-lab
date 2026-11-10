"""KV-cache inference for MiniGPT: pure-numpy forward (no autograd tape).

`prefill` runs a prompt once and stores per-layer keys/values; `step` then
processes one new token attending to the cache, so generating n tokens costs
O(n * T) attention work instead of recomputing the whole window each step.
Positions are learned/absolute, so once the sequence exceeds `ctx` the window
slides and the cache must be rebuilt: `generate_cached` falls back to a
re-prefill of the last ctx-1 tokens at that point (same result as `lm.generate`).
"""
import numpy as np


def _ln(ln, x):
    mu = x.mean(-1, keepdims=True)
    c = x - mu
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + ln.eps) * ln.g.data + ln.b.data


def _lin(l, x):
    return x @ l.w.data + l.b.data


def _gelu(v):
    c = np.sqrt(2 / np.pi)
    return 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v ** 3)))


class KVCache:
    def __init__(self, model):
        self.k = [None] * len(model.blocks)
        self.v = [None] * len(model.blocks)

    @property
    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[1]


    def append(self, i, k, v):
        """Store new (H, T, dh) keys/values for layer i; return the full (H, S, dh) arrays."""
        if self.k[i] is not None:
            k = np.concatenate([self.k[i], k], axis=1)
            v = np.concatenate([self.v[i], v], axis=1)
        self.k[i], self.v[i] = k, v
        return k, v

    def nbytes(self):
        return sum(a.nbytes for a in self.k + self.v if a is not None)


def _q8(x):
    """Symmetric int8 per (head, token) row: scale = max|x| / 127."""
    s = np.abs(x).max(-1, keepdims=True) / 127.0
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.rint(x / s), -127, 127).astype(np.int8), s.astype(np.float32)


class QuantKVCache(KVCache):
    """KV cache stored as int8 + one float32 scale per (head, token) vector.

    ~3.6x smaller than float64 at dh=16 (the scale overhead shrinks as dh grows).
    Attention reads dequantized values, so outputs differ slightly from exact.
    """

    def __init__(self, model):
        super().__init__(model)
        self.ks = [None] * len(model.blocks)
        self.vs = [None] * len(model.blocks)

    def append(self, i, k, v):
        qk, sk = _q8(k)
        qv, sv = _q8(v)
        if self.k[i] is not None:
            qk = np.concatenate([self.k[i], qk], axis=1)
            sk = np.concatenate([self.ks[i], sk], axis=1)
            qv = np.concatenate([self.v[i], qv], axis=1)
            sv = np.concatenate([self.vs[i], sv], axis=1)
        self.k[i], self.ks[i], self.v[i], self.vs[i] = qk, sk, qv, sv
        return qk * sk.astype(np.float64), qv * sv.astype(np.float64)

    def nbytes(self):
        return super().nbytes() + sum(a.nbytes for a in self.ks + self.vs if a is not None)


def _block(blk, x, cache, i):
    """x: (T_new, D) for one sequence; attends to cache + itself, causally."""
    a = blk.attn
    h = _ln(blk.ln1, x)
    T, D = h.shape
    qkv = _lin(a.qkv, h).reshape(T, 3, a.h, a.dh)
    q, k, v = (qkv[:, j].transpose(1, 0, 2) for j in range(3))  # (H, T, dh)
    k, v = cache.append(i, k, v)
    S = k.shape[1]
    att = q @ k.transpose(0, 2, 1) / np.sqrt(a.dh)
    # query t (absolute S-T+t) may see keys <= its own position
    mask = np.arange(S)[None, :] > (S - T + np.arange(T))[:, None]
    att = np.where(mask[None], -1e9, att)
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    y = (att @ v).transpose(1, 0, 2).reshape(T, D)
    x = x + _lin(a.proj, y)
    return x + _lin(blk.fc2, _gelu(_lin(blk.fc1, _ln(blk.ln2, x))))


def forward_new(model, ids, cache):
    """Process `ids` (1-D new tokens) on top of `cache`; return logits (T_new, V)."""
    start = cache.length
    ids = np.asarray(ids)
    assert start + len(ids) <= model.pos.data.shape[0], "sequence exceeds ctx"
    x = model.tok.data[ids] + model.pos.data[start:start + len(ids)]
    for i, blk in enumerate(model.blocks):
        x = _block(blk, x, cache, i)
    return _lin(model.head, _ln(model.lnf, x))


def prefill(model, ids, cache_cls=KVCache):
    cache = cache_cls(model)
    return forward_new(model, ids, cache)[-1], cache


def step(model, token, cache):
    return forward_new(model, [token], cache)[-1]


def _sample(lg, temperature, rng, top_k=None, top_p=None):
    """Temperature, then optional top-k, then optional top-p (nucleus) filtering.

    With top_k=top_p=None this consumes the rng exactly like the original
    sampler, so it still reproduces `lm.generate`.
    """
    lg = lg / temperature
    if top_k is not None and top_k < len(lg):
        kth = np.partition(lg, -top_k)[-top_k]
        lg = np.where(lg < kth, -np.inf, lg)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        csum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(csum, top_p) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(len(p), p=p))


def generate_cached(model, chars, prompt_ids, n, ctx, temperature=0.8, seed=0,
                    top_k=None, top_p=None, cache_cls=KVCache):
    """Same sampling stream as `lm.generate` (when top_k/top_p are None and the
    cache is exact), with cached decoding while it fits."""
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    lg, cache = prefill(model, ids[-ctx:], cache_cls)
    for _ in range(n):
        ids.append(_sample(lg, temperature, rng, top_k, top_p))
        if cache.length + 1 <= ctx:
            lg = step(model, ids[-1], cache)
        else:  # window slides: absolute positions shift, rebuild
            lg, cache = prefill(model, ids[-ctx:], cache_cls)
    return "".join(chars[i] for i in ids)


def cached_nll(model, ids, cache_cls=KVCache, window=None):
    """Mean next-token NLL (nats) of `ids`, teacher-forced one token at a time through the cache,
    so every attention read sees whatever the cache stored (exact or int8). Window length is
    model ctx unless `window` is smaller. Returns (nll, n_predictions)."""
    ids = np.asarray(ids)
    w = min(window or model.pos.data.shape[0], model.pos.data.shape[0])
    total, n = 0.0, 0
    for s in range(0, len(ids) - 1, w):
        chunk = ids[s:s + w + 1]
        if len(chunk) < 2:
            break
        cache = cache_cls(model)
        for j in range(len(chunk) - 1):
            lg = forward_new(model, [chunk[j]], cache)[-1]
            lg = lg - lg.max()
            total -= float(lg[chunk[j + 1]] - np.log(np.exp(lg).sum()))
            n += 1
    return total / n, n
