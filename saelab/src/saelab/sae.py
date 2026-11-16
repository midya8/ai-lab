"""Sparse autoencoder with hand-written backprop.

    pre = (x - b_dec) @ W_enc + b_enc
    a   = relu(pre)                       # 'relu' (sparsity from an L1 penalty)
    a   = top-k of relu(pre), rest zeroed # 'topk'  (sparsity from construction, k per sample)
    a   = top-(k*batch) of relu(pre) over the whole batch   # 'batchtopk' (k on average;
          samples may use more or fewer latents; inference uses a learned threshold)
    x^  = a @ W_dec + b_dec               # rows of W_dec are kept unit-norm

Loss = mean_n ||x - x^||^2  +  l1 * mean_n sum_j a_j.  Because decoder rows are unit-norm, the L1
term cannot be cheated by shrinking the decoder and growing the activations.
"""

from __future__ import annotations

import numpy as np


class SAE:
    def __init__(self, d: int, n_latents: int, mode: str = "relu", k: int | None = None,
                 l1: float = 0.0, seed: int = 0, dtype=np.float64):
        if mode not in ("relu", "topk", "batchtopk"):
            raise ValueError("mode must be 'relu', 'topk' or 'batchtopk'")
        if mode != "relu" and (k is None or not 1 <= k <= n_latents):
            raise ValueError("topk modes need 1 <= k <= n_latents")
        self.d, self.m, self.mode, self.k, self.l1 = d, n_latents, mode, k, l1
        self.threshold = 0.0   # batchtopk: inference cut-off, tracked as an EMA during training
        rng = np.random.default_rng(seed)
        wd = rng.standard_normal((n_latents, d)).astype(dtype)
        wd /= np.linalg.norm(wd, axis=1, keepdims=True)
        self.params = {
            "W_dec": wd,
            "W_enc": wd.T.copy(),           # tied initialisation
            "b_enc": np.zeros(n_latents, dtype),
            "b_dec": np.zeros(d, dtype),
        }

    # ------------------------------------------------------------------ forward
    def _select(self, act: np.ndarray, train: bool) -> np.ndarray:
        """Boolean mask of the latents that stay on (a constant in the backward pass)."""
        mask = act > 0
        if self.mode == "topk" and self.k < self.m:
            idx = np.argpartition(act, self.m - self.k, axis=1)[:, : self.m - self.k]
            mask = mask.copy()
            np.put_along_axis(mask, idx, False, axis=1)
        elif self.mode == "batchtopk":
            if train:
                keep = min(int(round(self.k * act.shape[0])), act.size)
                flat = act.ravel()
                cut = np.partition(flat, flat.size - keep)[flat.size - keep]
                mask = act >= max(cut, 1e-30)
                kept = act[mask]
                lo = float(kept.min()) if kept.size else 0.0
                self.threshold = lo if self.threshold == 0.0 else 0.99 * self.threshold + 0.01 * lo
            else:
                mask = act >= max(self.threshold, 1e-30)
        return mask

    def encode(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        act = np.maximum((x - p["b_dec"]) @ p["W_enc"] + p["b_enc"], 0.0)
        return act * self._select(act, train=False)

    def decode(self, a: np.ndarray) -> np.ndarray:
        return a @ self.params["W_dec"] + self.params["b_dec"]

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

    def loss_and_grads(self, x: np.ndarray):
        """Return ``(loss, grads, info)`` where grads has the same keys as ``params``."""
        p = self.params
        n = x.shape[0]
        xc = x - p["b_dec"]
        pre = xc @ p["W_enc"] + p["b_enc"]
        act = np.maximum(pre, 0.0)
        mask = self._select(act, train=True)
        a = act * mask
        xh = a @ p["W_dec"] + p["b_dec"]
        err = xh - x
        mse = float((err ** 2).sum(axis=1).mean())
        sparsity = float(a.sum(axis=1).mean())
        loss = mse + self.l1 * sparsity

        g = 2.0 * err / n
        grads = {"W_dec": a.T @ g, "b_dec": g.sum(axis=0)}
        da = g @ p["W_dec"].T + (self.l1 / n) * mask
        dpre = da * mask
        grads["W_enc"] = xc.T @ dpre
        grads["b_enc"] = dpre.sum(axis=0)
        grads["b_dec"] = grads["b_dec"] - (dpre @ p["W_enc"].T).sum(axis=0)
        return loss, grads, {"mse": mse, "l1_term": sparsity, "active": mask.sum(axis=1).mean(), "a": a}

    # ------------------------------------------------------------------ utilities
    def normalize_decoder(self) -> None:
        """Rescale each decoder row to unit norm and fold the scale into the encoder column."""
        wd = self.params["W_dec"]
        nrm = np.linalg.norm(wd, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        wd /= nrm
        self.params["W_enc"] *= nrm.T
        self.params["b_enc"] *= nrm[:, 0]

    def project_decoder_grad(self, grads: dict) -> None:
        """Drop the component of dW_dec parallel to each (unit) row, as in Anthropic's recipe."""
        wd = self.params["W_dec"]
        gd = grads["W_dec"]
        gd -= (gd * wd).sum(axis=1, keepdims=True) * wd

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.params.values())
