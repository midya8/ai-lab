"""Char-level language-model training on top of MiniGPT (autodiff from scratch)."""
import numpy as np
from .nn import softmax_cross_entropy
from .optim import Adam


def make_data(text, val_frac=0.15):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text])
    cut = int(len(ids) * (1 - val_frac))
    return chars, ids[:cut], ids[cut:]


def batches(ids, ctx, bs, rng):
    starts = rng.integers(0, len(ids) - ctx - 1, bs)
    x = np.stack([ids[s:s + ctx] for s in starts])
    y = np.stack([ids[s + 1:s + ctx + 1] for s in starts])
    return x, y


def lm_loss(model, x, y):
    logits = model(x)
    return softmax_cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def evaluate(model, ids, ctx, n=16, seed=123):
    rng = np.random.default_rng(seed)
    x, y = batches(ids, ctx, n, rng)
    return float(lm_loss(model, x, y).data)


def train(model, train_ids, val_ids, ctx, steps=300, bs=16, lr=3e-3, seed=0, log_every=50):
    rng = np.random.default_rng(seed)
    opt = Adam(model.parameters(), lr=lr)
    hist = []
    for s in range(1, steps + 1):
        x, y = batches(train_ids, ctx, bs, rng)
        opt.zero_grad()
        loss = lm_loss(model, x, y)
        loss.backward()
        opt.step()
        if s % log_every == 0 or s == 1:
            hist.append((s, float(loss.data), evaluate(model, val_ids, ctx)))
    return hist


def generate(model, chars, prompt_ids, n, ctx, temperature=0.8, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    for _ in range(n):
        w = np.array([ids[-ctx:]])
        lg = model(w).data[0, -1] / temperature
        p = np.exp(lg - lg.max()); p /= p.sum()
        ids.append(int(rng.choice(len(p), p=p)))
    return "".join(chars[i] for i in ids)
