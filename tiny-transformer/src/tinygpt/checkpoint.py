"""Self-contained checkpoints (config + tokenizer + weights) and perplexity eval."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .bpe import BPETokenizer
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def _tok_to_dict(tok) -> dict:
    if isinstance(tok, CharTokenizer):
        return {"type": "char", "chars": tok.chars}
    if isinstance(tok, BPETokenizer):
        return {"type": "bpe", "merges": [list(m) for m in tok.merges]}
    raise TypeError(f"unsupported tokenizer {type(tok).__name__}")


def _tok_from_dict(d: dict):
    if d["type"] == "char":
        t = CharTokenizer("")
        t.chars = list(d["chars"])
        t.stoi = {c: i for i, c in enumerate(t.chars)}
        return t
    if d["type"] == "bpe":
        return BPETokenizer([tuple(m) for m in d["merges"]])
    raise ValueError(f"unknown tokenizer type {d['type']!r}")


def save_checkpoint(path: str, model: GPT, tok) -> None:
    meta = json.dumps({"config": asdict(model.cfg), "tokenizer": _tok_to_dict(tok)})
    np.savez(path, __meta__=np.array(meta), **model.params)


def load_checkpoint(path: str):
    """Return (model, tokenizer) from a file written by save_checkpoint."""
    with np.load(path) as z:
        meta = json.loads(str(z["__meta__"]))
        model = GPT(GPTConfig(**meta["config"]))
        for k in model.params:
            model.params[k][...] = z[k]
    return model, _tok_from_dict(meta["tokenizer"])


def evaluate(model: GPT, data, batch: int = 16) -> dict:
    """Mean next-token cross-entropy over every non-overlapping window of `data`.

    Returns {"loss", "perplexity", "tokens"}. Deterministic (no sampling).
    """
    data = np.asarray(data)
    T = model.cfg.block_size
    n = (len(data) - 1) // T
    if n < 1:
        raise ValueError("data shorter than block_size + 1")
    xs = np.stack([data[i * T:(i + 1) * T] for i in range(n)])
    ys = np.stack([data[i * T + 1:(i + 1) * T + 1] for i in range(n)])
    total = 0.0
    for s in range(0, n, batch):
        x, y = xs[s:s + batch], ys[s:s + batch]
        _, loss, _ = model.forward(x, y)
        total += loss * x.size
    mean = total / xs.size
    return {"loss": mean, "perplexity": float(np.exp(mean)), "tokens": int(xs.size)}


def evaluate_by_position(model: GPT, data, bins: int = 4, batch: int = 16) -> dict:
    """Mean loss as a function of position inside the context window.

    Early positions see little context, so loss should fall with position when the
    model really uses its context. Returns {"per_position": [T losses],
    "bins": [{"start","end","loss"}], "gain": first_bin_loss - last_bin_loss}.
    """
    data = np.asarray(data)
    T = model.cfg.block_size
    n = (len(data) - 1) // T
    if n < 1:
        raise ValueError("data shorter than block_size + 1")
    if not 1 <= bins <= T:
        raise ValueError("bins must be in [1, block_size]")
    xs = np.stack([data[i * T:(i + 1) * T] for i in range(n)])
    ys = np.stack([data[i * T + 1:(i + 1) * T + 1] for i in range(n)])
    total = np.zeros(T)
    for s in range(0, n, batch):
        logits, _, _ = model.forward(xs[s:s + batch])
        z = logits - logits.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        y = ys[s:s + batch]
        nll = -np.take_along_axis(logp, y[..., None], -1)[..., 0]
        total += nll.sum(0)
    per = total / n
    edges = np.linspace(0, T, bins + 1).astype(int)
    out = [{"start": int(a), "end": int(b), "loss": float(per[a:b].mean())}
           for a, b in zip(edges[:-1], edges[1:])]
    return {"per_position": per.tolist(), "bins": out, "gain": out[0]["loss"] - out[-1]["loss"]}
