"""Sequential CNN container, softmax cross-entropy and a small-CNN builder."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .layers import BatchNorm2D, Conv2D, Dense, Dropout, Flatten, GlobalAvgPool, MaxPool2D, Normalize, ReLU


class Sequential:
    def __init__(self, layers, config=None):
        self.layers = list(layers)
        self.config = config
        self.acts: list = []     # output of every layer from the last forward
        self.gouts: list = []    # gradient w.r.t. every layer's output from the last backward

    # --- mode ---------------------------------------------------------
    def train(self):
        for layer in self.layers:
            layer.training = True
        return self

    def eval(self):
        for layer in self.layers:
            layer.training = False
        return self

    # --- passes -------------------------------------------------------
    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        self.acts = []
        for layer in self.layers:
            x = layer.forward(x)
            self.acts.append(x)
        return x

    def backward(self, g):
        self.gouts = [None] * len(self.layers)
        for i in range(len(self.layers) - 1, -1, -1):
            self.gouts[i] = g
            g = self.layers[i].backward(g)
        return g

    # --- parameters ---------------------------------------------------
    @property
    def dtype(self):
        for layer in self.layers:
            for p in layer.params().values():
                return p.dtype
        return np.float32

    def parameters(self):
        """List of (name, param, grad) with live, in-place-updatable arrays."""
        out = []
        for i, layer in enumerate(self.layers):
            g = layer.grads()
            for name, p in layer.params().items():
                out.append((f"{i}.{type(layer).__name__}.{name}", p, g[name]))
        return out

    def buffers(self):
        """List of (name, array) for non-trainable state such as BatchNorm running statistics."""
        out = []
        for i, layer in enumerate(self.layers):
            for name, b in layer.buffers().items():
                out.append((f"{i}.{type(layer).__name__}.{name}", b))
        return out

    def n_params(self) -> int:
        return int(sum(p.size for _, p, _ in self.parameters()))

    # --- inference ----------------------------------------------------
    def predict_logits(self, x, batch=256):
        was = [l.training for l in self.layers]
        self.eval()
        outs = [self.forward(x[i:i + batch]) for i in range(0, len(x), batch)]
        for layer, t in zip(self.layers, was):
            layer.training = t
        return np.concatenate(outs, axis=0)

    def predict_proba(self, x, batch=256):
        return softmax(self.predict_logits(x, batch))

    def predict(self, x, batch=256):
        return self.predict_logits(x, batch).argmax(axis=1)

    # --- persistence --------------------------------------------------
    def save(self, path):
        arrays = {name: p for name, p, _ in self.parameters()}
        arrays.update({name: b for name, b in self.buffers()})
        arrays["__config__"] = np.frombuffer(json.dumps(asdict(self.config)).encode(), dtype=np.uint8)
        np.savez(path, **arrays)

    @staticmethod
    def load(path):
        with np.load(path) as z:
            cfg = ModelConfig(**json.loads(bytes(z["__config__"]).decode()))
            model = build_cnn(cfg)
            for name, p, _ in model.parameters():
                p[...] = z[name]
            for name, b in model.buffers():
                b[...] = z[name]
        return model


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y, label_smoothing=0.0):
    """Mean softmax cross-entropy and its gradient w.r.t. the logits."""
    n, k = logits.shape
    p = softmax(logits)
    target = np.full((n, k), label_smoothing / k, dtype=p.dtype)
    target[np.arange(n), y] += 1.0 - label_smoothing
    loss = -(target * np.log(np.clip(p, 1e-12, None))).sum(axis=1).mean()
    return float(loss), (p - target) / n


@dataclass
class ModelConfig:
    n_classes: int = 6
    img: int = 24
    width: int = 8
    hidden: int = 64
    dropout: float = 0.0
    dtype: str = "float32"
    mean: float = 0.25
    std: float = 0.3
    seed: int = 0
    head: str = "dense"       # "dense": flatten -> hidden -> classes; "gap": third conv + global average pool
    batchnorm: bool = False


def build_cnn(cfg: ModelConfig | None = None, **overrides) -> Sequential:
    """conv-relu-pool x2 followed by a dense head or (``head="gap"``) a third conv + global
    average pool + linear classifier. ``batchnorm`` inserts BatchNorm2D after every conv.
    ``img`` must be divisible by 4."""
    cfg = cfg or ModelConfig(**overrides)
    if cfg.img % 4:
        raise ValueError("img must be divisible by 4")
    if cfg.head not in ("dense", "gap"):
        raise ValueError("head must be 'dense' or 'gap'")
    rng = np.random.default_rng(cfg.seed)
    dt = np.dtype(cfg.dtype)
    w = cfg.width
    side = cfg.img // 4

    def block(cin, cout, pool):
        out = [Conv2D(cin, cout, 3, 1, rng, dt)]
        if cfg.batchnorm:
            out.append(BatchNorm2D(cout, dtype=dt))
        out.append(ReLU())
        return out + ([MaxPool2D()] if pool else [])

    layers = [Normalize(cfg.mean, cfg.std)] + block(1, w, True) + block(w, 2 * w, True)
    if cfg.head == "gap":
        layers += block(2 * w, 4 * w, False)
        layers += [GlobalAvgPool(), Dropout(cfg.dropout, np.random.default_rng(cfg.seed + 1)),
                   Dense(4 * w, cfg.n_classes, rng, dt)]
    else:
        layers += [Flatten(), Dropout(cfg.dropout, np.random.default_rng(cfg.seed + 1)),
                   Dense(2 * w * side * side, cfg.hidden, rng, dt), ReLU(),
                   Dense(cfg.hidden, cfg.n_classes, rng, dt)]
    return Sequential(layers, cfg)
