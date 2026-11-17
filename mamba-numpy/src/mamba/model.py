"""A small Mamba-style language model with hand-written backprop.

Block (pre-norm, residual)::

    x -> RMSNorm -> in_proj -> (xs, z)
    xs -> causal depthwise conv -> SiLU -> SSM            (the sequence mixer)
    out = out_proj( SSM(.) * SiLU(z) )
    x <- x + out

``Mamba`` stacks blocks between an embedding and an output head. Parameters live in
``self.params()`` (a flat name -> array dict whose arrays are the layers' own, so
optimisers update them in place) and gradients come back from ``loss_and_grads``.
``step`` decodes one token with constant memory and matches the full forward exactly.
"""
from __future__ import annotations

import json

import numpy as np

from .layers import CausalDWConv, Linear, RMSNorm, SSM, silu, silu_backward


class Block:
    def __init__(self, d_model, d_state, rng, expand=2, d_conv=4, selective=True, scan_method="sequential"):
        self.d_inner = d_model * expand
        self.norm = RMSNorm(d_model)
        self.inp = Linear(d_model, 2 * self.d_inner, rng)
        self.conv = CausalDWConv(self.d_inner, d_conv, rng)
        self.ssm = SSM(self.d_inner, d_state, rng, selective=selective, scan_method=scan_method)
        self.out = Linear(self.d_inner, d_model, rng, scale=1.0 / np.sqrt(self.d_inner))
        self.parts = {"norm": self.norm, "inp": self.inp, "conv": self.conv, "ssm": self.ssm, "out": self.out}

    def forward(self, x):
        xz = self.inp.forward(self.norm.forward(x))
        xs, z = xz[..., : self.d_inner], xz[..., self.d_inner:]
        c = self.conv.forward(xs)
        a = silu(c)
        y = self.ssm.forward(a)
        gz = silu(z)
        self.cache = (c, z, y, gz)
        return x + self.out.forward(y * gz)

    def backward(self, dout):
        c, z, y, gz = self.cache
        dm = self.out.backward(dout)
        dy, dgz = dm * gz, dm * y
        dz = silu_backward(dgz, z)
        da = self.ssm.backward(dy)
        dxs = self.conv.backward(silu_backward(da, c))
        dxn = self.inp.backward(np.concatenate([dxs, dz], axis=-1))
        return dout + self.norm.backward(dxn)

    def init_state(self, batch):
        return {"h": self.ssm.init_state(batch), "buf": np.zeros((batch, self.conv.k - 1, self.d_inner))}

    def step(self, x, state):
        xz = self.inp.forward(self.norm.forward(x))
        xs, z = xz[:, : self.d_inner], xz[:, self.d_inner:]
        c, buf = self.conv.step(xs, state["buf"])
        y, h = self.ssm.step(silu(c), state["h"])
        return x + (y * silu(z)) @ self.out.p["W"], {"h": h, "buf": buf}


class Mamba:
    def __init__(self, vocab, d_model=32, d_state=8, n_layer=2, expand=2, d_conv=4, selective=True,
                 seed=0, scan_method="sequential"):
        rng = np.random.default_rng(seed)
        self.cfg = dict(vocab=vocab, d_model=d_model, d_state=d_state, n_layer=n_layer, expand=expand,
                        d_conv=d_conv, selective=selective)
        self.emb = rng.normal(0.0, 0.5, (vocab, d_model))
        self.blocks = [Block(d_model, d_state, rng, expand, d_conv, selective, scan_method) for _ in range(n_layer)]
        self.norm = RMSNorm(d_model)
        self.head = Linear(d_model, vocab, rng)

    # -- parameters -----------------------------------------------------
    def _layers(self):
        yield "norm_f", self.norm
        yield "head", self.head
        for i, b in enumerate(self.blocks):
            for name, layer in b.parts.items():
                yield f"blocks.{i}.{name}", layer

    def params(self):
        out = {"emb": self.emb}
        for prefix, layer in self._layers():
            for k, v in layer.p.items():
                out[f"{prefix}.{k}"] = v
        return out

    def grads(self):
        out = {"emb": self.demb}
        for prefix, layer in self._layers():
            for k, v in layer.g.items():
                out[f"{prefix}.{k}"] = v
        return out

    def n_params(self):
        return int(sum(v.size for v in self.params().values()))

    # -- forward / backward ---------------------------------------------
    def forward(self, tokens):
        self.tokens = tokens
        x = self.emb[tokens]
        for b in self.blocks:
            x = b.forward(x)
        return self.head.forward(self.norm.forward(x))

    def loss_and_grads(self, tokens, targets):
        """Mean cross-entropy over positions where ``targets >= 0``; fills ``grads()``."""
        logits = self.forward(tokens)
        mask = targets >= 0
        n = max(int(mask.sum()), 1)
        m = logits.max(-1, keepdims=True)
        lse = m[..., 0] + np.log(np.exp(logits - m).sum(-1))
        tgt = np.where(mask, targets, 0)
        logp = np.take_along_axis(logits, tgt[..., None], -1)[..., 0] - lse
        loss = -(logp * mask).sum() / n
        dlog = np.exp(logits - lse[..., None])
        np.put_along_axis(dlog, tgt[..., None], np.take_along_axis(dlog, tgt[..., None], -1) - 1.0, -1)
        dlog = dlog * mask[..., None] / n
        dx = self.norm.backward(self.head.backward(dlog))
        for b in reversed(self.blocks):
            dx = b.backward(dx)
        self.demb = np.zeros_like(self.emb)
        np.add.at(self.demb, self.tokens, dx)
        return float(loss), logits

    # -- constant-memory decoding -----------------------------------------
    def init_state(self, batch):
        return [b.init_state(batch) for b in self.blocks]

    def step(self, token, state):
        """token: (B,) ints. Returns (logits (B, V), new_state)."""
        x = self.emb[token]
        new = []
        for b, s in zip(self.blocks, state):
            x, s2 = b.step(x, s)
            new.append(s2)
        return self.norm.forward(x) @ self.head.p["W"], new

    def generate(self, prompt, n_new, temperature=0.0, seed=0):
        """Greedy (temperature 0) or sampled continuation of a 1-D prompt, one state update per token."""
        rng = np.random.default_rng(seed)
        state = self.init_state(1)
        out = list(prompt)
        logits = None
        for t in prompt:
            logits, state = self.step(np.array([t]), state)
        for _ in range(n_new):
            lg = logits[0]
            if temperature <= 0:
                nxt = int(np.argmax(lg))
            else:
                p = np.exp((lg - lg.max()) / temperature)
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            out.append(nxt)
            logits, state = self.step(np.array([nxt]), state)
        return out

    def state_bytes(self):
        """Bytes of recurrent state per sequence (independent of sequence length)."""
        c = self.cfg
        d_inner = c["d_model"] * c["expand"]
        return 8 * c["n_layer"] * (d_inner * c["d_state"] + (c["d_conv"] - 1) * d_inner)

    # -- checkpoints ------------------------------------------------------
    def save(self, path):
        np.savez(path, __cfg__=np.array(json.dumps(self.cfg)), **self.params())

    @classmethod
    def load(cls, path, scan_method="sequential"):
        with np.load(path) as z:
            cfg = json.loads(str(z["__cfg__"]))
            m = cls(**cfg, scan_method=scan_method)
            for k, v in m.params().items():
                if z[k].shape != v.shape:
                    raise ValueError(f"checkpoint tensor {k} has shape {z[k].shape}, model expects {v.shape}")
                v[...] = z[k]
        return m
