from __future__ import annotations

import time

import numpy as np

from .optim import AdamW, cosine_lr
from .tasks import score


def fit(model, make_batch, steps=1000, batch=32, lr=3e-3, seed=0, log_every=0, eval_batch=None, clip=1.0, wd=0.01):
    """Train on fresh samples every step. ``make_batch(n, rng) -> (tokens, targets)``.

    Returns a history list of dicts (step, loss, and eval token/exact accuracy every ``log_every`` steps).
    """
    rng = np.random.default_rng(seed)
    opt = AdamW(model.params(), lr=lr, clip=clip, wd=wd)
    hist = []
    ev = eval_batch if eval_batch is not None else make_batch(256, np.random.default_rng(10_000 + seed))
    t0 = time.time()
    for s in range(steps):
        toks, tgts = make_batch(batch, rng)
        loss, _ = model.loss_and_grads(toks, tgts)
        gn = opt.step(model.grads(), lr=cosine_lr(s, steps, lr))
        if log_every and (s % log_every == 0 or s == steps - 1):
            tok, exact = evaluate(model, ev)
            hist.append({"step": s, "loss": loss, "grad_norm": gn, "tok_acc": tok, "exact": exact,
                         "seconds": time.time() - t0})
    return hist


def evaluate(model, batch):
    toks, tgts = batch
    return score(model.forward(toks), tgts)
