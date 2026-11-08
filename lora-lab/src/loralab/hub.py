"""AdapterHub: many small LoRA adapters over one frozen tiny-transformer GPT.

One frozen base model, any number of named adapters, and three things you can do
with them that a single-adapter setup cannot:

* **swap** - activate one adapter by name (or none: bit-exact base model);
* **blend** - activate several at once with weights, ``W + sum_i w_i * delta_i``
  (blending the *deltas*, not the A/B factors, since ``mean(A) @ mean(B)`` is not
  ``mean(A @ B)``);
* **route** - pick the adapter whose NLL on a prompt is lowest, no labels needed;
* **compose** - fold a weighted blend into a *new* single adapter by truncated SVD
  of the blended delta (exact when the new rank >= sum of the parts' ranks).

All adapters in a hub share the same targets so they can be blended per weight.
"""
from __future__ import annotations

import contextlib
import json
import os

import numpy as np

from .gpt_lora import GPTLoRA

BASE = "<base>"


class AdapterHub:
    def __init__(self, model, targets=("qkv", "proj"), r=4, alpha=8.0):
        self.model, self.targets, self.r, self.alpha = model, tuple(targets), r, alpha
        self.adapters: dict[str, GPTLoRA] = {}

    # ---- registry ----
    def add(self, name, r=None, alpha=None, seed=0) -> GPTLoRA:
        if name == BASE or not name:
            raise ValueError(f"invalid adapter name {name!r}")
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already exists")
        lo = GPTLoRA(self.model, self.targets, r=r or self.r, alpha=alpha or self.alpha, seed=seed)
        self.adapters[name] = lo
        return lo

    def remove(self, name):
        del self.adapters[self._need(name)]

    def names(self):
        return sorted(self.adapters)

    def _need(self, name):
        if name not in self.adapters:
            raise KeyError(f"no adapter named {name!r}; have {self.names()}")
        return name

    # ---- activation ----
    def _weights(self, spec):
        """None -> base; 'a' -> {'a': 1}; {'a': .7, 'b': .3} -> as given."""
        if spec is None:
            return {}
        if isinstance(spec, str):
            spec = {spec: 1.0}
        return {self._need(k): float(v) for k, v in spec.items() if v != 0.0}

    def blended_delta(self, spec):
        w = self._weights(spec)
        out = {}
        for n in next(iter(self.adapters.values())).names if self.adapters else []:
            d = 0.0
            for k, wt in w.items():
                d = d + wt * self.adapters[k].delta(n)
            out[n] = d
        return out

    @contextlib.contextmanager
    def active(self, spec=None):
        """Temporarily load ``W + sum_i w_i * delta_i`` into the model."""
        w = self._weights(spec)
        if any(self.adapters[k].merged for k in w):
            raise RuntimeError("an adapter in this hub is merged into the model; unmerge it first")
        P = self.model.params
        saved = {}
        try:
            if w:
                for n, d in self.blended_delta(w).items():
                    saved[n] = P[n]
                    P[n] = P[n] + np.asarray(d).astype(P[n].dtype)
            yield self.model
        finally:
            for n, v in saved.items():
                P[n] = v

    # ---- evaluation / routing ----
    def evaluate_loss(self, data, block, spec=None, batches=20, batch=16, seed=0):
        rng = np.random.default_rng(seed)
        tot = 0.0
        with self.active(spec) as m:
            for _ in range(batches):
                ix = rng.integers(0, len(data) - block - 1, batch)
                x = np.stack([data[i:i + block] for i in ix])
                y = np.stack([data[i + 1:i + block + 1] for i in ix])
                tot += m.forward(x, y)[1]
        return tot / batches

    def prompt_nll(self, ids, spec=None, block=None):
        """Mean next-token NLL of ``ids`` under ``spec``, in non-overlapping windows."""
        ids = np.asarray(ids)
        block = block or self.model.cfg.block_size
        if len(ids) < 2:
            raise ValueError("need at least 2 tokens to score a prompt")
        total, count = 0.0, 0
        with self.active(spec) as m:
            for s in range(0, len(ids) - 1, block):
                x = ids[s:s + block][None]
                y = ids[s + 1:s + block + 1][None]
                x = x[:, :y.shape[1]]
                total += m.forward(x, y)[1] * y.shape[1]
                count += y.shape[1]
        return total / count

    def route(self, ids, candidates=None, include_base=True, block=None):
        """Return ``(best_name, {name: nll})``; ``best_name`` is BASE if the base wins."""
        names = list(candidates) if candidates is not None else self.names()
        scores = {n: self.prompt_nll(ids, n, block) for n in names}
        if include_base:
            scores[BASE] = self.prompt_nll(ids, None, block)
        return min(scores, key=scores.get), scores

    # ---- composition ----
    def compose(self, spec, name, rank=None):
        """Fold a weighted blend into a new adapter by per-weight truncated SVD.

        Returns ``(adapter, rel_error)`` where ``rel_error`` is the relative Frobenius
        error of the new adapter's delta against the exact blended delta (0 when the
        rank is large enough).
        """
        w = self._weights(spec)
        if not w:
            raise ValueError("compose needs at least one adapter with non-zero weight")
        rank = rank or sum(self.adapters[k].r for k in w)
        new = self.add(name, r=rank, alpha=self.alpha)
        exact = self.blended_delta(w)
        err_num = err_den = 0.0
        for n, D in exact.items():
            M = D / new.scale
            U, S, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
            k = min(rank, len(S))
            root = np.sqrt(S[:k])
            new.A[n][...] = 0.0
            new.B[n][...] = 0.0
            new.A[n][:, :k] = (U[:, :k] * root).astype(new.A[n].dtype)
            new.B[n][:k, :] = (root[:, None] * Vt[:k]).astype(new.B[n].dtype)
            err_num += float(np.sum((new.delta(n) - D) ** 2))
            err_den += float(np.sum(D ** 2))
        return new, (err_num / err_den) ** 0.5 if err_den > 0 else 0.0

    # ---- persistence ----
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        manifest = {"targets": list(self.targets), "r": self.r, "alpha": self.alpha, "adapters": {}}
        for name, lo in self.adapters.items():
            lo.save(os.path.join(directory, f"{name}.npz"))
            manifest["adapters"][name] = {"r": lo.r, "scale": lo.scale}
        with open(os.path.join(directory, "hub.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, model, directory):
        with open(os.path.join(directory, "hub.json")) as f:
            mf = json.load(f)
        hub = cls(model, tuple(mf["targets"]), r=mf["r"], alpha=mf["alpha"])
        for name, meta in mf["adapters"].items():
            lo = hub.add(name, r=meta["r"], alpha=meta["scale"] * meta["r"])
            lo.load(os.path.join(directory, f"{name}.npz"))
        return hub


def train_adapter(lora, data, steps, block, lr=1e-2, batch=16, seed=0):
    """Train one adapter on token array ``data`` with Adam; returns the loss curve."""
    from tinygpt import Adam
    rng = np.random.default_rng(seed)
    opt = Adam(lora.params(), lr=lr)
    curve = []
    for _ in range(steps):
        ix = rng.integers(0, len(data) - block - 1, batch)
        x = np.stack([data[i:i + block] for i in ix])
        y = np.stack([data[i + 1:i + block + 1] for i in ix])
        loss, grads = lora.loss_and_grads(x, y)
        opt.step(grads)
        curve.append(float(loss))
    return curve
