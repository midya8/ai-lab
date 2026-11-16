"""Kernel-regression view of gradient descent on a linearised network."""
from __future__ import annotations

import numpy as np


def stable_lr(K_train: np.ndarray, frac: float = 0.9) -> float:
    """Largest-ish stable GD step for loss (1/2n) sum r^2 on a model with train kernel K:
    stable iff lr * lambda_max(K) / n < 2."""
    n = K_train.shape[0]
    lam = np.linalg.eigvalsh(K_train)[-1]
    return float(2.0 * frac * n / lam)


def _g(lam: np.ndarray, lr: float, n: int, steps: int) -> np.ndarray:
    """(lr/n) * sum_{s<steps} (1 - lr*lam/n)^s, written to stay finite as lam -> 0."""
    a = lr * lam / n
    out = np.full_like(lam, lr * steps / n, dtype=float)
    ok = np.abs(a) > 1e-12
    out[ok] = (1.0 - (1.0 - a[ok]) ** steps) / lam[ok]
    return out


def gd_predict(K_train, K_test_train, f0_train, f0_test, y, lr: float, steps: int):
    """Exact function-space prediction after `steps` full-batch GD steps on
    L = (1/2n) sum (f(x_i) - y_i)^2 for the linearised model
    f(x) = f0(x) + J(x)(theta - theta0), whose kernel is K = J J^T.

    Returns (train_pred, test_pred).  K_test_train is K(x_test, X_train).
    """
    n = K_train.shape[0]
    lam, U = np.linalg.eigh(K_train)
    lam = np.clip(lam, 0.0, None)
    r0 = f0_train - y
    shrink = U @ (_g(lam, lr, n, steps) * (U.T @ r0))     # (lr/n) sum_s M^s r0
    # train: f_t = f0 - K * shrink  (K shrink == (I-M^t) r0)
    train = f0_train - K_train @ shrink
    test = f0_test - K_test_train @ shrink
    return train, test


def kernel_mean_prediction(K_train, K_test_train, y, ridge: float = 0.0):
    """Infinite-time (t -> inf) NTK-GP mean prediction with zero initial function:
    K_xX (K + ridge I)^-1 y."""
    n = K_train.shape[0]
    return K_test_train @ np.linalg.solve(K_train + ridge * np.eye(n), y)


def residual_modes(K_train, residual: np.ndarray):
    """Coefficients of a train-set residual in the eigenbasis of K (descending eigenvalue)."""
    lam, U = np.linalg.eigh(K_train)
    order = np.argsort(lam)[::-1]
    return lam[order], U[:, order], U[:, order].T @ residual
