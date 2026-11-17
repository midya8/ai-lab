"""LoRA on the attention (and optionally MLP) matrices of the tiny-transformer GPT.

The base GPT (``tinygpt``) is frozen. For every targeted weight ``W`` an adapter
``A (in, r)`` / ``B (r, out)`` is trained with ``W_eff = W + (alpha / r) * A @ B``.

No new backward pass is needed: the base model's hand-written backprop already
returns ``dL/dW_eff``, and the chain rule through ``W_eff`` gives

    dA = s * dW @ B.T          dB = s * A.T @ dW          (s = alpha / r)

so each step swaps ``W_eff`` into the model, runs the existing backward, restores
the frozen ``W`` and keeps only the adapter gradients.
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

# Cross-project import: tinygpt lives in the sibling ``tiny-transformer`` folder.
_TT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tiny-transformer", "src")
if _TT not in sys.path:
    sys.path.insert(0, _TT)

TARGETS = {"qkv": "qkv_W", "proj": "proj_W", "fc": "fc_W", "out": "out_W"}


class GPTLoRA:
    def __init__(self, model, targets=("qkv", "proj"), r=4, alpha=8.0, seed=0):
        unknown = [t for t in targets if t not in TARGETS]
        if unknown:
            raise ValueError(f"unknown LoRA targets {unknown}; choose from {sorted(TARGETS)}")
        self.model, self.r, self.scale = model, r, alpha / r
        self.targets = tuple(targets)
        rng = np.random.default_rng(seed)
        self.names = [f"h{i}.{TARGETS[t]}" for i in range(model.cfg.n_layer) for t in self.targets]
        dtype = next(iter(model.params.values())).dtype
        self.A, self.B = {}, {}
        for n in self.names:
            d_in, d_out = model.params[n].shape
            self.A[n] = (rng.standard_normal((d_in, r)) / np.sqrt(d_in)).astype(dtype)
            self.B[n] = np.zeros((r, d_out), dtype)  # zero-init => adapted model starts identical to base
        self.merged = False
        self.enabled = True

    # ---- parameters ----
    def params(self):
        """Trainable parameters as a flat dict (works with ``tinygpt.Adam``)."""
        p = {}
        for n in self.names:
            p[f"{n}.A"], p[f"{n}.B"] = self.A[n], self.B[n]
        return p

    def n_trainable(self) -> int:
        return int(sum(v.size for v in self.params().values()))

    def delta(self, name):
        return self.scale * (self.A[name] @ self.B[name])

    # ---- effective weights ----
    @contextlib.contextmanager
    def active(self, enabled=True):
        """Temporarily put ``W + delta`` (or plain ``W`` when disabled) into the model."""
        P = self.model.params
        saved = {n: P[n] for n in self.names}
        try:
            if enabled and not self.merged:
                for n in self.names:
                    P[n] = saved[n] + self.delta(n).astype(saved[n].dtype)
            yield self.model
        finally:
            for n, w in saved.items():
                P[n] = w

    def loss_and_grads(self, idx, targets, rng=None):
        """Loss plus gradients for the adapter parameters only."""
        assert not self.merged, "cannot train merged adapters; call unmerge() first"
        with self.active() as m:
            loss, G = m.loss_and_grads(idx, targets, rng=rng)
        out = {}
        for n in self.names:
            dW = G[n]
            out[f"{n}.A"] = self.scale * dW @ self.B[n].T
            out[f"{n}.B"] = self.scale * self.A[n].T @ dW
        return loss, out

    def forward(self, idx, targets=None):
        with self.active(self.enabled) as m:
            return m.forward(idx, targets)

    def evaluate_loss(self, data, block, batches=20, batch=16, seed=0, enabled=True):
        """Mean NLL over random windows of ``data`` with adapters on or off."""
        rng = np.random.default_rng(seed)
        tot = 0.0
        with self.active(enabled) as m:
            for _ in range(batches):
                ix = rng.integers(0, len(data) - block - 1, batch)
                x = np.stack([data[i:i + block] for i in ix])
                y = np.stack([data[i + 1:i + block + 1] for i in ix])
                tot += m.forward(x, y)[1]
        return tot / batches

    # ---- merge / unmerge ----
    def merge(self):
        """Fold adapters into the base weights: zero inference overhead."""
        if not self.merged:
            for n in self.names:
                self.model.params[n] += self.delta(n).astype(self.model.params[n].dtype)
            self.merged = True

    def unmerge(self):
        if self.merged:
            for n in self.names:
                self.model.params[n] -= self.delta(n).astype(self.model.params[n].dtype)
            self.merged = False

    # ---- adapter files ----
    def save(self, path):
        meta = {"r": np.array(self.r), "scale": np.array(self.scale), "targets": np.array(",".join(self.targets))}
        np.savez(path, **{f"A::{n}": a for n, a in self.A.items()}, **{f"B::{n}": b for n, b in self.B.items()}, **meta)

    def load(self, path):
        with np.load(path) as z:
            if int(z["r"]) != self.r or str(z["targets"]) != ",".join(self.targets):
                raise ValueError("adapter file does not match this GPTLoRA's rank/targets")
            for n in self.names:
                self.A[n][...] = z[f"A::{n}"]
                self.B[n][...] = z[f"B::{n}"]
