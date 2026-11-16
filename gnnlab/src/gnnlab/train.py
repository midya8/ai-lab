"""Full-batch Adam training with L2 weight decay and best-validation checkpointing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import GraphOps
from .metrics import accuracy
from .model import GNN, softmax_xent


@dataclass
class TrainResult:
    best_epoch: int
    train_acc: float
    val_acc: float
    test_acc: float
    losses: list = field(default_factory=list)


def fit(model: GNN, x, y, ops: GraphOps, train, val, test, epochs: int = 200,
        lr: float = 0.01, weight_decay: float = 5e-4, patience: int | None = None
        ) -> TrainResult:
    """Train in place, then restore the epoch with the best validation accuracy
    (ties broken by lower validation loss). Test accuracy is read only at that epoch."""
    m = {n: np.zeros_like(p) for n, p, _ in model.named_params()}
    v = {n: np.zeros_like(p) for n, p, _ in model.named_params()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = (-1.0, np.inf)
    best_state, best_epoch = model.state(), 0
    losses, since = [], 0
    for t in range(1, epochs + 1):
        logits = model.forward(x, ops)
        loss, dl = softmax_xent(logits, y, train)
        model.backward(dl)
        for n, p, g in model.named_params():
            if not n.endswith(".b"):
                g = g + weight_decay * p
            m[n] = b1 * m[n] + (1 - b1) * g
            v[n] = b2 * v[n] + (1 - b2) * g * g
            p -= lr * (m[n] / (1 - b1 ** t)) / (np.sqrt(v[n] / (1 - b2 ** t)) + eps)
        logits = model.forward(x, ops)
        vloss, _ = softmax_xent(logits, y, val)
        vacc = accuracy(logits, y, val)
        losses.append(loss)
        if (vacc, -vloss) > (best[0], -best[1]):
            best, best_state, best_epoch, since = (vacc, vloss), model.state(), t, 0
        else:
            since += 1
            if patience is not None and since >= patience:
                break
    model.load_state(best_state)
    logits = model.forward(x, ops)
    return TrainResult(best_epoch, accuracy(logits, y, train), accuracy(logits, y, val),
                       accuracy(logits, y, test), losses)
