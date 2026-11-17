"""Mini-batch trainer with augmentation, cosine LR, optional FGSM adversarial training and
optional interval-bound-propagation (IBP) training for certified robustness."""
from __future__ import annotations

import math
import time

import numpy as np

from .attacks import fgsm
from .certify import ibp_loss_backward
from .data import augment, photometric_augment
from .layers import BatchNorm2D
from .metrics import accuracy
from .model import cross_entropy
from .optim import Adam


def train(model, Xtr, ytr, Xva=None, yva=None, epochs=10, batch=64, lr=2e-3, weight_decay=1e-4,
          seed=0, aug=True, adv_eps=0.0, label_smoothing=0.0, ibp_eps=0.0, ibp_warmup=0.4,
          ibp_kappa=0.5, photometric=0.0, log=print):
    """Train in place; returns a list of per-epoch dicts (loss, train_acc, val_acc, seconds).

    adv_eps > 0 adds, for every batch, FGSM examples crafted against the current model
    (Goodfellow et al.), so the loss is over clean + adversarial images.

    ibp_eps > 0 trains for *certified* robustness (Gowal et al., 2018): the loss is
    ``kappa * CE(clean) + (1 - kappa) * CE(worst case over the eps-box, from interval bounds)``.
    eps ramps linearly from 0 to ibp_eps and kappa from 1 to ibp_kappa over the first ``ibp_warmup``
    fraction of the steps, then both stay put. Not available with BatchNorm2D or together with adv_eps.
    """
    # photometric > 0: per-image contrast augmentation, c ~ U(1 - photometric, 1) (see data.photometric_augment)
    if ibp_eps > 0 and adv_eps > 0:
        raise ValueError("choose either adv_eps (FGSM) or ibp_eps (interval bound propagation)")
    if ibp_eps > 0 and any(isinstance(l, BatchNorm2D) for l in model.layers):
        raise ValueError("ibp_eps is not supported with BatchNorm2D")
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
            if photometric > 0:
                xb = photometric_augment(xb, rng, 1.0 - photometric)
            if adv_eps > 0:
                xb = np.concatenate([xb, fgsm(model, xb, yb, adv_eps)])
                yb = np.concatenate([yb, yb])
            opt.lr = lr * 0.5 * (1 + math.cos(math.pi * step / total))
            model.train()
            logits = model.forward(xb)
            loss, dl = cross_entropy(logits, yb, label_smoothing)
            if ibp_eps > 0:
                ramp = min(1.0, step / max(1.0, ibp_warmup * total))
                eps_t, kappa = ibp_eps * ramp, 1.0 - (1.0 - ibp_kappa) * ramp
                model.backward(dl * kappa)
                if eps_t > 0:
                    saved = [g.copy() for _, _, g in model.parameters()]
                    iloss, _ = ibp_loss_backward(model, xb, yb, eps_t, 1.0 - kappa, label_smoothing)
                    for (_, _, g), g0 in zip(model.parameters(), saved):
                        g += g0
                    loss = kappa * loss + (1.0 - kappa) * iloss
            else:
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
