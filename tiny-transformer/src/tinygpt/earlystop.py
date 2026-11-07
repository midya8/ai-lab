"""Early stopping on validation loss with best-weights restore."""
from __future__ import annotations

import math


class EarlyStopper:
    """Track validation loss; stop after `patience` evaluations without
    an improvement of at least `min_delta`. Keeps a copy of the best params."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.best_step = None
        self.bad = 0
        self._snap = None

    def update(self, val_loss: float, params: dict, step: int) -> bool:
        """Record an evaluation. Returns True when training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best, self.best_step, self.bad = val_loss, step, 0
            self._snap = {k: v.copy() for k, v in params.items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, params: dict) -> bool:
        """Copy best weights back in place. False if nothing was recorded."""
        if self._snap is None:
            return False
        for k, v in self._snap.items():
            params[k][...] = v
        return True
