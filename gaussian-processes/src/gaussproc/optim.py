"""A compact L-BFGS minimiser (two-loop recursion, Armijo backtracking, box clipping)
so that hyper-parameter fitting needs nothing beyond NumPy."""
from __future__ import annotations

from typing import Callable

import numpy as np


def lbfgs(fun: Callable[[np.ndarray], tuple[float, np.ndarray]], x0: np.ndarray,
          max_iter: int = 100, m: int = 10, gtol: float = 1e-5,
          bounds: tuple[float, float] | None = (-9.0, 9.0)) -> tuple[np.ndarray, float, int]:
    """Minimise ``fun`` (returns value and gradient). Returns (x, f, iterations).

    Non-finite function values are treated as +inf so a line search backs off from them.
    """
    lo, hi = bounds if bounds is not None else (-np.inf, np.inf)
    x = np.clip(np.asarray(x0, float).copy(), lo, hi)

    def safe(z):
        try:
            f, g = fun(z)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(z)
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            return np.inf, np.zeros_like(z)
        return float(f), np.asarray(g, float)

    f, g = safe(x)
    if not np.isfinite(f):
        return x, f, 0
    S: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    it = 0
    for it in range(1, max_iter + 1):
        # gradient of the clipped problem: zero components pushing against an active bound
        gp = g.copy()
        gp[(x <= lo + 1e-12) & (g > 0)] = 0.0
        gp[(x >= hi - 1e-12) & (g < 0)] = 0.0
        if np.max(np.abs(gp)) < gtol:
            break
        q = gp.copy()
        alphas = []
        for s, y in zip(reversed(S), reversed(Y)):
            rho = 1.0 / float(y @ s)
            a = rho * float(s @ q)
            alphas.append((a, rho, s, y))
            q -= a * y
        if S:
            gamma = float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
        else:
            gamma = 1.0 / max(np.linalg.norm(gp), 1.0)
        r = gamma * q
        for a, rho, s, y in reversed(alphas):
            b = rho * float(y @ r)
            r += s * (a - b)
        d = -r
        if float(d @ gp) >= 0:          # not a descent direction: reset memory
            S.clear(); Y.clear()
            d = -gp / max(np.linalg.norm(gp), 1.0)
        t = 1.0
        ok = False
        for _ in range(40):
            xn = np.clip(x + t * d, lo, hi)
            fn, gn = safe(xn)
            if fn <= f + 1e-4 * float(gp @ (xn - x)):
                ok = True
                break
            t *= 0.5
        if not ok:
            if S:
                S.clear(); Y.clear()
                continue
            break
        s_vec, y_vec = xn - x, gn - g
        if float(s_vec @ y_vec) > 1e-12 * np.linalg.norm(s_vec) * np.linalg.norm(y_vec):
            S.append(s_vec); Y.append(y_vec)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        converged = abs(f - fn) < 1e-12 * max(1.0, abs(f))
        x, f, g = xn, fn, gn
        if converged:
            break
    return x, f, it
