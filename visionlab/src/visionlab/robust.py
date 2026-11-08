"""Robustness report: natural perturbations and white-box attacks."""
from __future__ import annotations

import numpy as np

from .attacks import fgsm, pgd
from .metrics import accuracy


def shift(X, dy, dx):
    """Translate with zero fill (no wrap-around)."""
    out = np.zeros_like(X)
    H, W = X.shape[-2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd] = X[..., ys, xs]
    return out


def natural_report(model, X, y, seed=0, noise=(0.0, 0.1, 0.2, 0.3, 0.4), shifts=(0, 2, 4), contrasts=(1.0, 0.7, 0.5, 0.3)):
    rng = np.random.default_rng(seed)
    rep = {"noise": {}, "shift": {}, "contrast": {}}
    for s in noise:
        Xn = np.clip(X + rng.normal(0, s, X.shape), 0, 1).astype(X.dtype) if s else X
        rep["noise"][s] = accuracy(y, model.predict(Xn))
    for k in shifts:
        rep["shift"][k] = accuracy(y, model.predict(shift(X, k, k)))
    for c in contrasts:
        rep["contrast"][c] = accuracy(y, model.predict(np.clip(X * c, 0, 1).astype(X.dtype)))
    return rep


def attack_report(model, X, y, eps=(0.0, 0.02, 0.05, 0.1, 0.2), pgd_steps=10, batch=200, seed=0):
    """Accuracy under FGSM and PGD at each L-inf budget (eps in [0,1] pixel units)."""
    rng = np.random.default_rng(seed)
    rep = {"fgsm": {}, "pgd": {}}
    for e in eps:
        hit_f = hit_p = 0
        for s in range(0, len(X), batch):
            xb, yb = X[s:s + batch], y[s:s + batch]
            if e == 0:
                xf = xp = xb
            else:
                xf = fgsm(model, xb, yb, e)
                xp = pgd(model, xb, yb, e, steps=pgd_steps, rng=rng)
            hit_f += int((model.predict(xf) == yb).sum())
            hit_p += int((model.predict(xp) == yb).sum())
        rep["fgsm"][e] = hit_f / len(X)
        rep["pgd"][e] = hit_p / len(X)
    return rep


def format_table(rep, title=""):
    lines = [title] if title else []
    for section, vals in rep.items():
        lines.append(f"{section:<8}" + "  ".join(f"{k:>6}" for k in vals))
        lines.append(f"{'acc':<8}" + "  ".join(f"{v:>6.3f}" for v in vals.values()))
    return "\n".join(lines)
