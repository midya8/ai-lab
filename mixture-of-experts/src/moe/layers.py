"""Dense FFN and sparse MoE layers with hand-written backward passes.

MoELayer routing modes
----------------------
``topk``          token-choice: every token picks its ``k`` best experts; an expert accepts at
                  most ``capacity`` tokens, the overflow is *dropped* (the residual stream
                  carries the token past the layer untouched).
``expert_choice`` expert-choice (Zhou et al. 2022): every expert picks its ``capacity``
                  best tokens, so load is perfectly balanced by construction but a token may
                  get no expert at all, or several.

Two DeepSeek-style extras (v0.2), both optional and off by default:

``n_shared``      always-on shared experts whose outputs are added to every token (gate 1, not
                  subject to capacity), so routed experts only need to learn what is *not*
                  common to all tokens.
``bias_rate``     auxiliary-loss-free balancing (Wang et al. 2024): a per-expert bias is added to
                  the router logits for *selection only* (gate values still come from the
                  unbiased softmax) and nudged by ``bias_rate`` after each step against the
                  sign of that expert's load error.  No balance-loss gradient touches the router.

Discrete decisions (which experts, which tokens survive the capacity limit) are treated as
constants in the backward pass; gradients flow through the continuous gate weights, which is
what real implementations do.  The finite-difference tests rely on routing being locally
stable under a 1e-6 perturbation.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Experts:
    """``E`` independent two-layer ReLU MLPs ``h -> f -> h`` stored as stacked arrays."""

    def __init__(self, n: int, d: int, f: int, rng: np.random.Generator):
        self.W1 = rng.normal(size=(n, d, f)) * np.sqrt(2.0 / d)
        self.b1 = np.zeros((n, f))
        self.W2 = rng.normal(size=(n, f, d)) * np.sqrt(1.0 / f) * 0.5
        self.b2 = np.zeros((n, d))

    def forward(self, e: int, x: np.ndarray):
        a = x @ self.W1[e] + self.b1[e]
        h = np.maximum(a, 0.0)
        return h @ self.W2[e] + self.b2[e], (x, a, h)

    def backward(self, e: int, cache, dy: np.ndarray, grads: dict):
        x, a, h = cache
        grads["W2"][e] += h.T @ dy
        grads["b2"][e] += dy.sum(axis=0)
        da = (dy @ self.W2[e].T) * (a > 0)
        grads["W1"][e] += x.T @ da
        grads["b1"][e] += da.sum(axis=0)
        return da @ self.W1[e].T


class DenseFFN:
    """One wide FFN with the same interface as :class:`MoELayer` (residual included)."""

    def __init__(self, d: int, f: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.ex = Experts(1, d, f, rng)
        self.aux_loss = 0.0
        self.last_stats: dict = {}

    def params(self) -> dict:
        return {"W1": self.ex.W1, "b1": self.ex.b1, "W2": self.ex.W2, "b2": self.ex.b2}

    def n_params(self) -> int:
        return sum(p.size for p in self.params().values())

    def forward(self, h: np.ndarray):
        y, c = self.ex.forward(0, h)
        self.aux_loss = 0.0
        return h + y, c

    def backward(self, dout: np.ndarray, cache):
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh = dout + self.ex.backward(0, cache, dout, grads)
        return dh, grads


class MoELayer:
    def __init__(self, d: int, f: int, n_experts: int = 8, k: int = 2,
                 capacity_factor: float | None = 1.25, routing: str = "topk",
                 aux_coef: float = 0.0, z_coef: float = 0.0, renorm: bool | None = None,
                 jitter: float = 0.0, seed: int = 0, n_shared: int = 0,
                 bias_rate: float = 0.0):
        if routing not in ("topk", "expert_choice"):
            raise ValueError("routing must be 'topk' or 'expert_choice'")
        if not 1 <= k <= n_experts:
            raise ValueError("need 1 <= k <= n_experts")
        if bias_rate and routing != "topk":
            raise ValueError("bias_rate (aux-loss-free balancing) needs routing='topk'")
        if n_shared < 0 or bias_rate < 0:
            raise ValueError("n_shared and bias_rate must be >= 0")
        self.d, self.f, self.E, self.k = d, f, n_experts, k
        self.n_shared, self.bias_rate = n_shared, bias_rate
        self.cf, self.routing = capacity_factor, routing
        self.aux_coef, self.z_coef, self.jitter = aux_coef, z_coef, jitter
        # k = 1 with renormalised gates would be a constant 1: the router would get no task gradient.
        self.renorm = (k > 1) if renorm is None else renorm
        rng = np.random.default_rng(seed)
        self.Wg = rng.normal(size=(d, n_experts)) * 0.02
        self.ex = Experts(n_experts, d, f, rng)
        # created after the routed experts so n_shared=0 reproduces v0.1 initialisation exactly
        self.sh = Experts(n_shared, d, f, rng) if n_shared else None
        self.bias = np.zeros(n_experts)          # selection bias; not a gradient parameter
        self._rng = np.random.default_rng(seed + 1)
        self.aux_loss = 0.0
        self.last_stats: dict = {}
        self.train_mode = True

    # ------------------------------------------------------------------ params
    def params(self) -> dict:
        p = {"Wg": self.Wg, "W1": self.ex.W1, "b1": self.ex.b1,
             "W2": self.ex.W2, "b2": self.ex.b2}
        if self.sh is not None:
            p.update({"sW1": self.sh.W1, "sb1": self.sh.b1, "sW2": self.sh.W2, "sb2": self.sh.b2})
        return p

    def n_params(self) -> int:
        return sum(p.size for p in self.params().values())

    def active_params_per_token(self) -> int:
        per_expert = self.d * self.f * 2 + self.f + self.d
        return self.d * self.E + (self.k + self.n_shared) * per_expert

    def update_bias(self) -> None:
        """One aux-loss-free balancing step from the last training forward's routing load.

        ``bias_e += rate * sign(mean_load - load_e)``: overloaded experts become less likely to
        be selected, starved ones more likely.  Uses the pre-drop selection share (what the
        router *wanted*), as in DeepSeek-V3.  No-op when ``bias_rate == 0``.
        """
        if not self.bias_rate or "load" not in self.last_stats:
            return
        load = self.last_stats["load"]
        self.bias += self.bias_rate * np.sign(load.mean() - load)

    def capacity(self, n_tokens: int) -> int:
        if self.cf is None:
            return n_tokens
        return max(1, int(np.ceil(self.cf * n_tokens * self.k / self.E)))

    # ----------------------------------------------------------------- forward
    def forward(self, h: np.ndarray):
        N = h.shape[0]
        jit = None
        hin = h
        if self.jitter and self.train_mode:
            jit = self._rng.uniform(1 - self.jitter, 1 + self.jitter, size=h.shape)
            hin = h * jit
        logits = hin @ self.Wg
        p = _softmax(logits)
        C = self.capacity(N)
        if self.routing == "topk":
            asg = self._route_topk(p, C, logits + self.bias if self.bias_rate else None)
        else:
            asg = self._route_expert_choice(p, C)
        # asg: boolean (N, E) of *kept* token->expert assignments
        # gate weights g (N, E), zero where not kept
        if self.routing == "topk":
            sel = asg["selected"]
            S = (p * sel).sum(axis=1, keepdims=True) if self.renorm else np.ones((N, 1))
            g = p * sel / S
            g = g * asg["kept"]
        else:
            S = np.ones((N, 1))
            g = p * asg["kept"]
        out = h.copy()
        scache = []
        if self.sh is not None:
            for s_ in range(self.n_shared):
                y, c = self.sh.forward(s_, h)
                out += y
                scache.append(c)
        ecache = {}
        eout = {}
        for e in range(self.E):
            idx = np.nonzero(asg["kept"][:, e])[0]
            if idx.size == 0:
                continue
            y, c = self.ex.forward(e, h[idx])
            ecache[e] = (idx, c)
            eout[e] = y
            out[idx] += g[idx, e:e + 1] * y

        # ---- auxiliary losses (Switch Transformer load balance + ST-MoE router z-loss)
        top1 = np.argmax(p, axis=1)
        f_frac = np.bincount(top1, minlength=self.E) / N       # hard, constant
        P_mean = p.mean(axis=0)
        lb = self.E * float(f_frac @ P_mean)
        lse = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
        zl = float(np.mean(lse ** 2))
        self.aux_loss = self.aux_coef * lb + self.z_coef * zl

        slots = asg["selected"].sum(axis=0)
        n_sel = max(int(asg["selected"].sum()), 1)
        self.last_stats = {
            "load": slots / max(slots.sum(), 1),          # share of routing slots per expert (pre-drop)
            "kept_load": asg["kept"].sum(axis=0) / max(asg["kept"].sum(), 1),
            "top1": top1,
            "drop_rate": float(1.0 - asg["kept"].sum() / n_sel),
            "token_unserved": float(np.mean(asg["kept"].sum(axis=1) == 0)),
            "router_entropy": float(np.mean(-(p * np.log(p + 1e-12)).sum(1))),
            "lb": lb, "z": zl, "capacity": C,
        }
        cache = {"h": h, "hin": hin, "p": p, "g": g, "S": S, "asg": asg, "ecache": ecache,
                 "eout": eout, "scache": scache, "jit": jit, "f_frac": f_frac, "lse": lse, "logits": logits}
        return out, cache

    def _route_topk(self, p: np.ndarray, C: int, score: np.ndarray | None = None) -> dict:
        N = p.shape[0]
        order = np.argsort(-(p if score is None else score), axis=1, kind="stable")[:, : self.k]
        selected = np.zeros_like(p, dtype=bool)
        np.put_along_axis(selected, order, True, axis=1)
        kept = np.zeros_like(selected)
        # priority: all first choices before any second choice, then token order (as in GShard)
        for rank in range(self.k):
            counts = kept.sum(axis=0)
            for n in range(N):
                e = order[n, rank]
                if counts[e] < C:
                    kept[n, e] = True
                    counts[e] += 1
        return {"selected": selected, "kept": kept}

    def _route_expert_choice(self, p: np.ndarray, C: int) -> dict:
        N = p.shape[0]
        C = min(C, N)
        kept = np.zeros_like(p, dtype=bool)
        top = np.argsort(-p, axis=0, kind="stable")[:C]          # (C, E) token ids per expert
        for e in range(self.E):
            kept[top[:, e], e] = True
        return {"selected": kept.copy(), "kept": kept}

    # ---------------------------------------------------------------- backward
    def backward(self, dout: np.ndarray, cache):
        h, hin, p, g, S = cache["h"], cache["hin"], cache["p"], cache["g"], cache["S"]
        asg, N = cache["asg"], h.shape[0]
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh = dout.copy()                                            # residual path
        dg = np.zeros_like(p)
        if self.sh is not None:
            sg = {"W1": np.zeros_like(self.sh.W1), "b1": np.zeros_like(self.sh.b1),
                  "W2": np.zeros_like(self.sh.W2), "b2": np.zeros_like(self.sh.b2)}
            for s_, c in enumerate(cache["scache"]):
                dh += self.sh.backward(s_, c, dout, sg)
            for name, v in sg.items():
                grads["s" + name] = v
        for e, (idx, c) in cache["ecache"].items():
            y = cache["eout"][e]
            dg[idx, e] = (dout[idx] * y).sum(axis=1)
            dh[idx] += self.ex.backward(e, c, dout[idx] * g[idx, e:e + 1], grads)
        # gate -> softmax probs
        if self.routing == "topk" and self.renorm:
            sel = asg["selected"]
            # g_i = p_i * kept_i / S  with S = sum_selected p ; kept mask is constant
            mask_k = asg["kept"]
            dg_k = dg * mask_k
            term = (dg_k * p).sum(axis=1, keepdims=True) / (S ** 2)
            dp = (dg_k / S - term) * sel
        elif self.routing == "topk":
            dp = dg * asg["kept"]
        else:
            dp = dg * asg["kept"]
        # auxiliary losses act directly on p / logits
        dp = dp + self.aux_coef * self.E * cache["f_frac"][None, :] / N
        dlogits = p * (dp - (dp * p).sum(axis=1, keepdims=True))
        if self.z_coef:
            dlogits = dlogits + self.z_coef * 2.0 * cache["lse"][:, None] * p / N
        grads["Wg"] += hin.T @ dlogits
        dgate = dlogits @ self.Wg.T
        dh += dgate if cache["jit"] is None else dgate * cache["jit"]
        return dh, grads
