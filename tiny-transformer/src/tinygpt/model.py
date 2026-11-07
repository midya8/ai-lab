"""A GPT (decoder-only transformer) in pure NumPy with hand-written backprop.

Pre-LayerNorm blocks, causal multi-head attention, GELU MLP, learned positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_C = np.sqrt(2.0 / np.pi)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0  # applied to embeddings, attention probs and both residual branches (train only)


# ---- primitives (each returns output + cache; *_bwd returns input grad) ----
def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * rstd
    return xh * g + b, (xh, rstd, g)


def ln_bwd(dy, cache):
    xh, rstd, g = cache
    dxh = dy * g
    dx = rstd * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, (dy * xh).reshape(-1, xh.shape[-1]).sum(0), dy.reshape(-1, xh.shape[-1]).sum(0)


def lin_fwd(x, W, b):
    return x @ W + b, x


def lin_bwd(dy, x, W):
    d = x.shape[-1]
    return dy @ W.T, x.reshape(-1, d).T @ dy.reshape(-1, dy.shape[-1]), dy.reshape(-1, dy.shape[-1]).sum(0)


def gelu_fwd(x):
    t = np.tanh(_C * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t), (x, t)


def gelu_bwd(dy, cache):
    x, t = cache
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * _C * (1 + 3 * 0.044715 * x * x))


def drop_fwd(x, p, rng):
    """Inverted dropout. Returns (y, mask); mask is None when inactive."""
    if not p or rng is None:
        return x, None
    mask = (rng.random(x.shape) >= p).astype(x.dtype) / (1.0 - p)
    return x * mask, mask


def drop_bwd(dy, mask):
    return dy if mask is None else dy * mask


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class GPT:
    def __init__(self, cfg: GPTConfig, seed: int = 0, dtype=np.float32):
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        C, V = cfg.n_embd, cfg.vocab_size
        n = lambda *s, std=0.02: (rng.standard_normal(s) * std).astype(dtype)
        z = lambda *s: np.zeros(s, dtype)
        o = lambda *s: np.ones(s, dtype)
        p = {"wte": n(V, C), "wpe": n(cfg.block_size, C, std=0.01),
             "lnf_g": o(C), "lnf_b": z(C), "head_W": n(C, V), "head_b": z(V)}
        proj_std = 0.02 / np.sqrt(2 * cfg.n_layer)
        for i in range(cfg.n_layer):
            p.update({f"h{i}.ln1_g": o(C), f"h{i}.ln1_b": z(C),
                      f"h{i}.qkv_W": n(C, 3 * C), f"h{i}.qkv_b": z(3 * C),
                      f"h{i}.proj_W": n(C, C, std=proj_std), f"h{i}.proj_b": z(C),
                      f"h{i}.ln2_g": o(C), f"h{i}.ln2_b": z(C),
                      f"h{i}.fc_W": n(C, 4 * C), f"h{i}.fc_b": z(4 * C),
                      f"h{i}.out_W": n(4 * C, C, std=proj_std), f"h{i}.out_b": z(C)})
        self.params = p

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ---- attention ----
    def _attn_fwd(self, x, i, rng=None):
        P, H = self.params, self.cfg.n_head
        B, T, C = x.shape
        D = C // H
        qkv, xin = lin_fwd(x, P[f"h{i}.qkv_W"], P[f"h{i}.qkv_b"])
        q, k, v = (t.reshape(B, T, H, D).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, -1))
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D)
        s = np.where(np.tril(np.ones((T, T), bool)), s, -1e30)
        a = softmax(s)
        ad, ma = drop_fwd(a, self.cfg.dropout, rng)
        y = (ad @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        out, yin = lin_fwd(y, P[f"h{i}.proj_W"], P[f"h{i}.proj_b"])
        return out, (xin, q, k, v, a, ad, ma, yin)

    def _attn_bwd(self, dout, cache, i, G):
        P, H = self.params, self.cfg.n_head
        xin, q, k, v, a, ad, ma, yin = cache
        B, T, C = xin.shape
        D = C // H
        dy, G[f"h{i}.proj_W"], G[f"h{i}.proj_b"] = lin_bwd(dout, yin, P[f"h{i}.proj_W"])
        do = dy.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        da = do @ v.transpose(0, 1, 3, 2)
        dv = ad.transpose(0, 1, 3, 2) @ do
        da = drop_bwd(da, ma)
        ds = a * (da - (da * a).sum(-1, keepdims=True)) / np.sqrt(D)
        dq, dk = ds @ k, ds.transpose(0, 1, 3, 2) @ q
        m = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqkv = np.concatenate([m(dq), m(dk), m(dv)], -1)
        dx, G[f"h{i}.qkv_W"], G[f"h{i}.qkv_b"] = lin_bwd(dqkv, xin, P[f"h{i}.qkv_W"])
        return dx

    # ---- full model ----
    def forward(self, idx, targets=None, rng=None):
        """idx: (B,T) ints. Returns logits, loss (None if no targets), cache.

        Pass an `rng` to enable dropout (training); without one the pass is deterministic.
        """
        P, cfg = self.params, self.cfg
        B, T = idx.shape
        assert T <= cfg.block_size
        p_drop = cfg.dropout
        x, me = drop_fwd(P["wte"][idx] + P["wpe"][:T], p_drop, rng)
        caches = []
        for i in range(cfg.n_layer):
            h, c1 = ln_fwd(x, P[f"h{i}.ln1_g"], P[f"h{i}.ln1_b"])
            a, ca = self._attn_fwd(h, i, rng)
            a, m1 = drop_fwd(a, p_drop, rng)
            x = x + a
            h, c2 = ln_fwd(x, P[f"h{i}.ln2_g"], P[f"h{i}.ln2_b"])
            f, cf = lin_fwd(h, P[f"h{i}.fc_W"], P[f"h{i}.fc_b"])
            g, cg = gelu_fwd(f)
            m, cm = lin_fwd(g, P[f"h{i}.out_W"], P[f"h{i}.out_b"])
            m, m2 = drop_fwd(m, p_drop, rng)
            x = x + m
            caches.append((c1, ca, c2, cf, cg, cm, m1, m2))
        h, cl = ln_fwd(x, P["lnf_g"], P["lnf_b"])
        logits, hin = lin_fwd(h, P["head_W"], P["head_b"])
        loss = None
        if targets is not None:
            pr = softmax(logits)
            loss = float(-np.log(pr[np.arange(B)[:, None], np.arange(T), targets] + 1e-30).mean())
        return logits, loss, (idx, targets, caches, cl, hin, logits, me)

    def backward(self, cache):
        P, cfg = self.params, self.cfg
        idx, targets, caches, cl, hin, logits, me = cache
        B, T = idx.shape
        G = {}
        d = softmax(logits)
        d[np.arange(B)[:, None], np.arange(T), targets] -= 1
        d /= B * T
        dh, G["head_W"], G["head_b"] = lin_bwd(d, hin, P["head_W"])
        dx, G["lnf_g"], G["lnf_b"] = ln_bwd(dh, cl)
        dx = dx.astype(dh.dtype)
        for i in reversed(range(cfg.n_layer)):
            c1, ca, c2, cf, cg, cm, m1, m2 = caches[i]
            dg, G[f"h{i}.out_W"], G[f"h{i}.out_b"] = lin_bwd(drop_bwd(dx, m2), cm, P[f"h{i}.out_W"])
            df = gelu_bwd(dg, cg)
            dh, G[f"h{i}.fc_W"], G[f"h{i}.fc_b"] = lin_bwd(df, cf, P[f"h{i}.fc_W"])
            dl, G[f"h{i}.ln2_g"], G[f"h{i}.ln2_b"] = ln_bwd(dh, c2)
            dx = dx + dl
            dh = self._attn_bwd(drop_bwd(dx, m1), ca, i, G)
            dl, G[f"h{i}.ln1_g"], G[f"h{i}.ln1_b"] = ln_bwd(dh, c1)
            dx = dx + dl
        dx = drop_bwd(dx, me)
        G["wpe"] = np.zeros_like(P["wpe"])
        G["wpe"][:T] = dx.sum(0)
        G["wte"] = np.zeros_like(P["wte"])
        np.add.at(G["wte"], idx, dx)
        return G

    def loss_and_grads(self, idx, targets, rng=None):
        _, loss, cache = self.forward(idx, targets, rng=rng)
        return loss, self.backward(cache)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, seed=0, top_p=None):
        from .sampling import sample
        rng = np.random.default_rng(seed)
        idx = np.array(idx, dtype=np.int64).reshape(1, -1)
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits = self.forward(ctx)[0][0, -1].astype(np.float64)
            nxt = sample(logits, rng, temperature, top_k, top_p)
            idx = np.concatenate([idx, [[nxt]]], 1)
        return idx[0]

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        with np.load(path) as z:
            for k in self.params:
                self.params[k][...] = z[k]
