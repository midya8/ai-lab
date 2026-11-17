"""A readable NumPy forward pass of a microgrid MiniGPT with hook points and an activation cache.

Everything the circuit analyses need (per-head outputs, attention patterns, the residual
stream) is exposed. `from_minigpt` copies weights from a trained microgrid model, and the tests
check the logits agree to float64 precision, so analyses on this class are analyses of the
trained model, not of a re-implementation.

Hook points (names): "embed", "L{l}.pattern" (B,H,T,T), "L{l}.head_out" (B,H,T,D),
"L{l}.resid_mid" (B,T,D, after attention), "L{l}.mlp_hidden" (B,T,4D, post-GELU), "L{l}.mlp_out" (B,T,D).
A hook is a callable array -> array (or None to leave it unchanged).
"""
import numpy as np


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    c = x - mu
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + eps) * g + b


def _gelu(v):
    c = np.sqrt(2 / np.pi)
    return 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v ** 3)))


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class InterpModel:
    def __init__(self, params, heads):
        self.p = params
        self.H = heads
        self.layers = len(params["blocks"])
        self.D = params["tok"].shape[1]
        self.dh = self.D // heads
        self.V = params["tok"].shape[0]

    @classmethod
    def from_minigpt(cls, m):
        blocks = []
        for b in m.blocks:
            blocks.append(dict(
                ln1=(b.ln1.g.data.copy(), b.ln1.b.data.copy()), ln2=(b.ln2.g.data.copy(), b.ln2.b.data.copy()),
                qkv=(b.attn.qkv.w.data.copy(), b.attn.qkv.b.data.copy()),
                proj=(b.attn.proj.w.data.copy(), b.attn.proj.b.data.copy()),
                fc1=(b.fc1.w.data.copy(), b.fc1.b.data.copy()), fc2=(b.fc2.w.data.copy(), b.fc2.b.data.copy())))
        params = dict(tok=m.tok.data.copy(), pos=m.pos.data.copy(), blocks=blocks,
                      lnf=(m.lnf.g.data.copy(), m.lnf.b.data.copy()),
                      head=(m.head.w.data.copy(), m.head.b.data.copy()))
        return cls(params, m.blocks[0].attn.h)

    def save(self, path):
        flat = {"heads": np.array(self.H), "tok": self.p["tok"], "pos": self.p["pos"],
                "lnf_g": self.p["lnf"][0], "lnf_b": self.p["lnf"][1],
                "head_w": self.p["head"][0], "head_b": self.p["head"][1], "layers": np.array(self.layers)}
        for i, b in enumerate(self.p["blocks"]):
            for k, (w, bb) in b.items():
                flat[f"b{i}_{k}_0"], flat[f"b{i}_{k}_1"] = w, bb
        np.savez(path, **flat)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        blocks = [{k: (z[f"b{i}_{k}_0"], z[f"b{i}_{k}_1"]) for k in ("ln1", "ln2", "qkv", "proj", "fc1", "fc2")}
                  for i in range(int(z["layers"]))]
        params = dict(tok=z["tok"], pos=z["pos"], blocks=blocks, lnf=(z["lnf_g"], z["lnf_b"]),
                      head=(z["head_w"], z["head_b"]))
        return cls(params, int(z["heads"]))

    # ------------------------------------------------------------------ forward
    def forward(self, ids, hooks=None, cache=None):
        """Logits (B,T,V). `cache` (a dict) is filled with activations when given."""
        hooks = hooks or {}
        ids = np.asarray(ids)
        B, T = ids.shape
        H, dh, D = self.H, self.dh, self.D

        def hook(name, x):
            if cache is not None:
                cache[name] = x
            fn = hooks.get(name)
            if fn is not None:
                y = fn(x)
                if y is not None:
                    x = y
                    if cache is not None:
                        cache[name] = x
            return x

        resid = hook("embed", self.p["tok"][ids] + self.p["pos"][:T])
        mask = np.triu(np.ones((T, T), bool), 1)
        for l, blk in enumerate(self.p["blocks"]):
            if cache is not None:
                cache[f"L{l}.resid_pre"] = resid
            x = _ln(resid, *blk["ln1"])
            qkv = (x @ blk["qkv"][0] + blk["qkv"][1]).reshape(B, T, 3, H, dh)
            q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))  # (B,H,T,dh)
            att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
            att = np.where(mask, -1e9, att)
            pat = hook(f"L{l}.pattern", _softmax(att))
            z = pat @ v  # (B,H,T,dh)
            wo = blk["proj"][0].reshape(H, dh, D)
            head_out = hook(f"L{l}.head_out", np.einsum("bhtd,hde->bhte", z, wo))
            resid = hook(f"L{l}.resid_mid", resid + head_out.sum(1) + blk["proj"][1])
            m = _ln(resid, *blk["ln2"])
            hid = hook(f"L{l}.mlp_hidden", _gelu(m @ blk["fc1"][0] + blk["fc1"][1]))
            mlp = hook(f"L{l}.mlp_out", hid @ blk["fc2"][0] + blk["fc2"][1])
            resid = resid + mlp
        if cache is not None:
            cache["resid_final"] = resid
        return _ln(resid, *self.p["lnf"]) @ self.p["head"][0] + self.p["head"][1]

    def log_probs(self, ids, hooks=None):
        lg = self.forward(ids, hooks)
        lg = lg - lg.max(-1, keepdims=True)
        return lg - np.log(np.exp(lg).sum(-1, keepdims=True))
