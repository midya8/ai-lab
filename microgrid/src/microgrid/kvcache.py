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


def _block(blk, x, cache, i):
    """x: (T_new, D) for one sequence; attends to cache + itself, causally."""
    a = blk.attn
    h = _ln(blk.ln1, x)
    T, D = h.shape
    qkv = _lin(a.qkv, h).reshape(T, 3, a.h, a.dh)
    q, k, v = (qkv[:, j].transpose(1, 0, 2) for j in range(3))  # (H, T, dh)
    if cache.k[i] is not None:
        k = np.concatenate([cache.k[i], k], axis=1)
        v = np.concatenate([cache.v[i], v], axis=1)
    cache.k[i], cache.v[i] = k, v
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


def prefill(model, ids):
    cache = KVCache(model)
    return forward_new(model, ids, cache)[-1], cache


def step(model, token, cache):
    return forward_new(model, [token], cache)[-1]


def _sample(lg, temperature, rng):
    lg = lg / temperature
    p = np.exp(lg - lg.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_cached(model, chars, prompt_ids, n, ctx, temperature=0.8, seed=0):
    """Same sampling stream as `lm.generate`, with cached decoding while it fits."""
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    lg, cache = prefill(model, ids[-ctx:])
    for _ in range(n):
        ids.append(_sample(lg, temperature, rng))
        if cache.length + 1 <= ctx:
            lg = step(model, ids[-1], cache)
        else:  # window slides: absolute positions shift, rebuild
            lg, cache = prefill(model, ids[-ctx:])
    return "".join(chars[i] for i in ids)
