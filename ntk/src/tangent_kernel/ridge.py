"""Kernel ridge regression with closed-form leave-one-out (LOO) model selection.

For f = K (K + lam I)^-1 y the hat matrix is H = K (K + lam I)^-1 and the LOO residual of
point i is (y_i - f_i) / (1 - H_ii) -- no refitting needed.  One eigendecomposition of K
gives every lambda on a grid.
"""
from __future__ import annotations

import numpy as np


def loo_curve(K: np.ndarray, y: np.ndarray, lams):
    """Return (loo_mse, train_mse) arrays over the ridge grid `lams`."""
    lam_k, U = np.linalg.eigh(K)
    lam_k = np.clip(lam_k, 0.0, None)
    Uy = U.T @ y
    loo, tr = [], []
    for lam in lams:
        s = lam_k / (lam_k + lam)                      # eigenvalues of H
        f = U @ (s * Uy)
        h = (U ** 2) @ s                               # diag(H)
        res = y - f
        loo.append(np.mean((res / (1.0 - h)) ** 2))
        tr.append(np.mean(res ** 2))
    return np.array(loo), np.array(tr)


def select_ridge(K, y, lams) -> float:
    loo, _ = loo_curve(K, y, lams)
    return float(np.asarray(lams)[int(np.argmin(loo))])


def ridge_predict(K, K_test_train, y, lam: float):
    return K_test_train @ np.linalg.solve(K + lam * np.eye(len(y)), y)


def ridge_study(n: int = 40, noise: float = 0.3, seeds: int = 20, depth: int = 2):
    """On noisy circle data: test MSE of interpolation (lam~0), the LOO-chosen ridge,
    and the oracle-best ridge on the grid."""
    from .analytic import ntk
    lams = np.logspace(-6, 2, 33)
    phi_t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    Xt = np.stack([np.cos(phi_t), np.sin(phi_t)], 1)
    ft = np.sin(phi_t) + 0.5 * np.sin(3 * phi_t)
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        phi = rng.uniform(0, 2 * np.pi, n)
        X = np.stack([np.cos(phi), np.sin(phi)], 1)
        y = np.sin(phi) + 0.5 * np.sin(3 * phi) + noise * rng.normal(size=n)
        K, Kt = ntk(X, depth=depth), ntk(Xt, X, depth=depth)
        errs = np.array([np.mean((ridge_predict(K, Kt, y, l) - ft) ** 2) for l in lams])
        lam_loo = select_ridge(K, y, lams)
        rows.append((errs[0], errs[int(np.argmin(np.abs(lams - lam_loo)))], errs.min(), lam_loo))
    r = np.array(rows)
    return {"interp": r[:, 0].mean(), "loo": r[:, 1].mean(), "oracle": r[:, 2].mean(),
            "median_lam_loo": float(np.median(r[:, 3]))}


if __name__ == "__main__":
    for n in (20, 40, 80):
        print(n, ridge_study(n=n))
