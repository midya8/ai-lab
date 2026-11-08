"""Mini-batch trainer with augmentation, cosine LR and optional FGSM adversarial training."""
from __future__ import annotations

import math
import time

import numpy as np

from .attacks import fgsm
from .data import augment
from .metrics import accuracy
from .model import cross_entropy
from .optim import Adam


def train(model, Xtr, ytr, Xva=None, yva=None, epochs=10, batch=64, lr=2e-3, weight_decay=1e-4,
          seed=0, aug=True, adv_eps=0.0, label_smoothing=0.0, log=print):
    """Train in place; returns a list of per-epoch dicts (loss, train_acc, val_acc, seconds).

    adv_eps > 0 adds, for every batch, FGSM examples crafted against the current model
    (Goodfellow et al.), so the loss is over clean + adversarial images.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model, lr=lr, weight_decay=weight_decay)
    n = len(Xtr)
    steps_per = math.ceil(n / batch)
    total = epochs * steps_per
    step = 0
    history = []
    for ep in range(epochs):
        t0 = time.time()
        order = rng.permutation(n)
        losses, hits = [], 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = Xtr[idx], ytr[idx]
            if aug:
                xb = augment(xb, rng)
            if adv_eps > 0:
                xb = np.concatenate([xb, fgsm(model, xb, yb, adv_eps)])
                yb = np.concatenate([yb, yb])
            opt.lr = lr * 0.5 * (1 + math.cos(math.pi * step / total))
            model.train()
            logits = model.forward(xb)
            loss, dl = cross_entropy(logits, yb, label_smoothing)
            model.backward(dl)
            opt.step()
            losses.append(loss)
            hits += int((logits.argmax(1) == yb).sum())
            step += 1
        rec = {"epoch": ep + 1, "loss": float(np.mean(losses)),
               "train_acc": hits / (n * (2 if adv_eps > 0 else 1)), "seconds": time.time() - t0}
        if Xva is not None:
            rec["val_acc"] = accuracy(yva, model.predict(Xva))
        history.append(rec)
        if log:
            va = f" val_acc={rec['val_acc']:.3f}" if "val_acc" in rec else ""
            log(f"epoch {ep + 1:>2}/{epochs} loss={rec['loss']:.4f} train_acc={rec['train_acc']:.3f}{va} ({rec['seconds']:.1f}s)")
    model.eval()
    return history
