"""Two more SAE architectures with hand-written backprop: JumpReLU and Gated.

Both share the ``SAE`` interface (``params``, ``loss_and_grads``, ``encode``, ``decode``,
``normalize_decoder``, ``project_decoder_grad``) so ``fit`` and every metric work unchanged.

JumpReLU (Rajamanoharan et al., 2024):  a = pre * [pre > theta]  with a learned per-latent theta,
trained on  MSE + lam * mean_n sum_j [pre_j > theta_j]  (an L0 penalty). The step function has no
gradient, so theta gets a straight-through pseudo-gradient from a rectangle kernel of width ``eps``
around the threshold:  d[pre>theta]/dtheta ~ -K((pre-theta)/eps)/eps  and
d(pre*[pre>theta])/dtheta ~ -(theta/eps) K((pre-theta)/eps).  The encoder gets the usual gradient
through the active set only (the L0 term does not train the encoder directly).

Gated SAE (Rajamanoharan et al., 2024):  a gate decides *which* latents fire, a separate magnitude
path decides *how much*,  f = [gate_pre > 0] * relu(mag_pre),  with the magnitude weights tied to
the gate's, W_mag = W_gate * exp(r_mag), so the L1 penalty only has to act on the gate and does not
shrink magnitudes. Loss = MSE(f) + l1 * sum relu(gate_pre) + MSE(x, relu(gate_pre) W_dec + b_dec)
where the auxiliary reconstruction treats the decoder as a constant (stop-gradient).
"""

from __future__ import annotations

import numpy as np

from .sae import SAE


class _Base:
    """Shared decoder utilities. Subclasses define ``params`` including W_dec and b_dec."""

    def decode(self, a: np.ndarray) -> np.ndarray:
        return a @ self.params["W_dec"] + self.params["b_dec"]

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

    def project_decoder_grad(self, grads: dict) -> None:
        wd = self.params["W_dec"]
        gd = grads["W_dec"]
        gd -= (gd * wd).sum(axis=1, keepdims=True) * wd

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.params.values())

    def _unit_decoder(self):
        wd = self.params["W_dec"]
        nrm = np.maximum(np.linalg.norm(wd, axis=1, keepdims=True), 1e-12)
        wd /= nrm
        return nrm[:, 0]


class JumpReLUSAE(_Base):
    def __init__(self, d: int, n_latents: int, l0_coef: float = 0.01, eps: float = 0.05,
                 theta_init: float = 0.15, seed: int = 0, dtype=np.float64):
        if eps <= 0 or theta_init <= 0:
            raise ValueError("eps and theta_init must be positive")
        self.d, self.m, self.l0_coef, self.eps = d, n_latents, l0_coef, eps
        rng = np.random.default_rng(seed)
        wd = rng.standard_normal((n_latents, d)).astype(dtype)
        wd /= np.linalg.norm(wd, axis=1, keepdims=True)
        self.params = {"W_dec": wd, "W_enc": wd.T.copy(), "b_enc": np.zeros(n_latents, dtype),
                       "b_dec": np.zeros(d, dtype), "theta": np.full(n_latents, theta_init, dtype)}

    def encode(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        pre = (x - p["b_dec"]) @ p["W_enc"] + p["b_enc"]
        return pre * (pre > p["theta"])

    def loss_and_grads(self, x: np.ndarray):
        p, n = self.params, x.shape[0]
        xc = x - p["b_dec"]
        pre = xc @ p["W_enc"] + p["b_enc"]
        on = pre > p["theta"]
        a = pre * on
        xh = a @ p["W_dec"] + p["b_dec"]
        err = xh - x
        mse = float((err ** 2).sum(axis=1).mean())
        l0 = float(on.sum(axis=1).mean())
        loss = mse + self.l0_coef * l0

        g = 2.0 * err / n
        grads = {"W_dec": a.T @ g, "b_dec": g.sum(axis=0)}
        da = g @ p["W_dec"].T
        dpre = da * on
        grads["W_enc"] = xc.T @ dpre
        grads["b_enc"] = dpre.sum(axis=0)
        grads["b_dec"] = grads["b_dec"] - (dpre @ p["W_enc"].T).sum(axis=0)
        near = (np.abs(pre - p["theta"]) < self.eps / 2).astype(pre.dtype) / self.eps   # K(u)/eps
        grads["theta"] = -(da * p["theta"] * near).sum(axis=0) - (self.l0_coef / n) * near.sum(axis=0)
        return loss, grads, {"mse": mse, "l0_term": l0, "active": float(on.sum(axis=1).mean()), "a": a}

    def normalize_decoder(self) -> None:
        nrm = self._unit_decoder()
        self.params["W_enc"] *= nrm[None, :]
        self.params["b_enc"] *= nrm
        self.params["theta"] *= nrm
        np.maximum(self.params["theta"], 1e-6, out=self.params["theta"])   # thresholds stay positive


class GatedSAE(_Base):
    def __init__(self, d: int, n_latents: int, l1: float = 0.2, seed: int = 0, dtype=np.float64):
        self.d, self.m, self.l1 = d, n_latents, l1
        rng = np.random.default_rng(seed)
        wd = rng.standard_normal((n_latents, d)).astype(dtype)
        wd /= np.linalg.norm(wd, axis=1, keepdims=True)
        self.params = {"W_dec": wd, "W_gate": wd.T.copy(), "b_gate": np.zeros(n_latents, dtype),
                       "r_mag": np.zeros(n_latents, dtype), "b_mag": np.zeros(n_latents, dtype),
                       "b_dec": np.zeros(d, dtype)}

    def _forward(self, xc):
        p = self.params
        u = xc @ p["W_gate"] + p["b_gate"]
        v = xc @ (p["W_gate"] * np.exp(p["r_mag"])[None, :]) + p["b_mag"]
        return u, v

    def encode(self, x: np.ndarray) -> np.ndarray:
        u, v = self._forward(x - self.params["b_dec"])
        return (u > 0) * np.maximum(v, 0.0)

    def _loss_value(self, x, aux_W, aux_b):
        """Loss with the auxiliary term's decoder given explicitly (a constant, as stop-gradient)."""
        p = self.params
        u, v = self._forward(x - p["b_dec"])
        f = (u > 0) * np.maximum(v, 0.0)
        mse = ((f @ p["W_dec"] + p["b_dec"] - x) ** 2).sum(axis=1).mean()
        g = np.maximum(u, 0.0)
        aux = ((g @ aux_W + aux_b - x) ** 2).sum(axis=1).mean()
        return float(mse + self.l1 * g.sum(axis=1).mean() + aux)

    def loss_and_grads(self, x: np.ndarray):
        p, n = self.params, x.shape[0]
        xc = x - p["b_dec"]
        u, v = self._forward(xc)
        gate = u > 0
        mag = np.maximum(v, 0.0)
        f = gate * mag
        xh = f @ p["W_dec"] + p["b_dec"]
        err = xh - x
        g = np.maximum(u, 0.0)
        xa = g @ p["W_dec"] + p["b_dec"]          # the decoder is a constant in this term
        err_a = xa - x
        mse = float((err ** 2).sum(axis=1).mean())
        aux = float((err_a ** 2).sum(axis=1).mean())
        sparse = float(g.sum(axis=1).mean())
        loss = mse + self.l1 * sparse + aux

        dxh = 2.0 * err / n
        grads = {"W_dec": f.T @ dxh, "b_dec": dxh.sum(axis=0)}
        dv = (dxh @ p["W_dec"].T) * gate * (v > 0)
        escale = np.exp(p["r_mag"])
        w_mag = p["W_gate"] * escale[None, :]
        dw_mag = xc.T @ dv
        grads["b_mag"] = dv.sum(axis=0)
        grads["r_mag"] = (dw_mag * w_mag).sum(axis=0)
        dxc = dv @ w_mag.T
        dg = self.l1 / n + (2.0 * err_a / n) @ p["W_dec"].T
        du = dg * gate
        grads["W_gate"] = dw_mag * escale[None, :] + xc.T @ du
        grads["b_gate"] = du.sum(axis=0)
        dxc = dxc + du @ p["W_gate"].T
        grads["b_dec"] = grads["b_dec"] - dxc.sum(axis=0)
        return loss, grads, {"mse": mse, "aux": aux, "l1_term": sparse, "active": float(gate.sum(axis=1).mean()),
                             "a": f}

    def normalize_decoder(self) -> None:
        nrm = self._unit_decoder()
        # f scales with the magnitude path: W_mag columns and b_mag by nrm; W_mag = W_gate * exp(r)
        self.params["r_mag"] += np.log(nrm)
        self.params["b_mag"] *= nrm
