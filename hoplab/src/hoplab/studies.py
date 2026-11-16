"""Reproducible studies. Every function returns plain dicts/lists; ``python -m hoplab <study>`` prints a table."""
from __future__ import annotations

import numpy as np

from . import classical as C
from . import modern as M
from .glyphs import digit_patterns, ROWS, COLS, render
from .patterns import random_patterns, correlated_patterns, flip_noise, mask_noise

RULES = ("hebbian", "pseudoinverse", "storkey", "dam-poly3", "dam-poly5", "dam-exp")


def _recall(rule: str, X: np.ndarray, Q: np.ndarray, rng: np.random.Generator, beta: float = 12.0):
    """Run retrieval of rule from queries Q; returns final states."""
    if rule == "hebbian":
        return C.retrieve(C.hebbian(X), Q, rng=rng).states
    if rule == "pseudoinverse":
        return C.retrieve(C.pseudoinverse(X), Q, rng=rng).states
    if rule == "storkey":
        return C.retrieve(C.storkey(X), Q, rng=rng).states
    if rule == "dam-poly3":
        return M.dam_retrieve(X, Q, kind="poly", order=3, rng=rng).states
    if rule == "dam-poly5":
        return M.dam_retrieve(X, Q, kind="poly", order=5, rng=rng).states
    if rule == "dam-exp":
        return M.dam_retrieve(X, Q, kind="exp", beta=beta, rng=rng).states
    raise ValueError(rule)


def exact_recovery(rule: str, X: np.ndarray, noise: float, rng: np.random.Generator, queries: int | None = None) -> float:
    """Fraction of (a sample of) stored patterns recovered *exactly* from a `noise`-flipped copy."""
    p = X.shape[0]
    idx = np.arange(p) if queries is None or queries >= p else rng.choice(p, size=queries, replace=False)
    Q = flip_noise(X[idx], noise, rng)
    out = _recall(rule, X, Q, rng)
    return float(np.mean(np.all(out == X[idx], axis=1)))


def capacity_study(n: int = 100, ps=(5, 10, 14, 20, 30, 50, 100, 200, 400), noise: float = 0.10,
                   seeds: int = 3, queries: int = 30, rules=RULES, rho: float = 0.0):
    """Exact-recovery rate vs load alpha = p/n for each storage rule (random or rho-correlated patterns)."""
    rows = []
    for p in ps:
        row = {"p": p, "alpha": p / n}
        for rule in rules:
            vals = []
            for s in range(seeds):
                rng = np.random.default_rng(1000 * s + p)
                X = random_patterns(p, n, rng) if rho == 0 else correlated_patterns(p, n, rho, rng)
                vals.append(exact_recovery(rule, X, noise, rng, queries))
            row[rule] = float(np.mean(vals))
        rows.append(row)
    return rows


def basin_study(n: int = 100, p: int = 20, noises=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4), seeds: int = 3,
                queries: int = 20, rules=("hebbian", "pseudoinverse", "storkey", "dam-poly3", "dam-exp")):
    """How much corruption each memory tolerates at a fixed load (basin-of-attraction radius)."""
    rows = []
    for noise in noises:
        row = {"noise": noise}
        for rule in rules:
            vals = []
            for s in range(seeds):
                rng = np.random.default_rng(7 + 31 * s)
                X = random_patterns(p, n, rng)
                vals.append(exact_recovery(rule, X, noise, rng, queries))
            row[rule] = float(np.mean(vals))
        rows.append(row)
    return rows


def correlated_study(n: int = 100, p: int = 20, rhos=(0.0, 0.2, 0.4, 0.6), noise: float = 0.05, seeds: int = 3,
                     queries: int = 20, rules=("hebbian", "pseudoinverse", "storkey", "dam-poly3", "dam-exp")):
    """Same load, but patterns share a template: pairwise overlap rho^2."""
    rows = []
    for rho in rhos:
        row = {"rho": rho}
        for rule in rules:
            vals = []
            for s in range(seeds):
                rng = np.random.default_rng(99 + 17 * s)
                X = correlated_patterns(p, n, rho, rng)
                vals.append(exact_recovery(rule, X, noise, rng, queries))
            row[rule] = float(np.mean(vals))
        rows.append(row)
    return rows


def beta_study(d: int = 64, p: int = 32, sigma: float = 0.5, betas=(0.1, 1.0, 3.0, 6.0, 9.0, 12.0, 20.0, 50.0),
               seeds: int = 5, queries: int = 32, rho: float = 0.0):
    """Continuous Hopfield: inverse temperature beta vs retrieval accuracy and the number of patterns the fixed point mixes.

    Patterns are unit-norm; the query is a stored pattern plus N(0, sigma^2/d) noise per coordinate.
    Low beta -> the fixed point is (close to) the mean of all patterns; high beta -> a single pattern.
    """
    rows = []
    for beta in betas:
        acc, eff, steps, e1 = [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(5 + s)
            X = rng.normal(size=(p, d))
            if rho:
                X = np.sqrt(1 - rho) * X + np.sqrt(rho) * rng.normal(size=(1, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            idx = rng.choice(p, size=min(queries, p), replace=False)
            Q = X[idx] + rng.normal(size=(len(idx), d)) * (sigma / np.sqrt(d))
            res = M.cont_retrieve(X, Q, beta, max_steps=200, tol=1e-10)
            nearest = np.argmax(res.states @ X.T, axis=1)
            close = np.linalg.norm(res.states - X[nearest], axis=1) < 0.05
            acc.append(np.mean((nearest == idx) & close))
            eff.append(np.mean(M.effective_number(res.weights)))
            steps.append(np.mean(res.steps))
            one, _ = M.cont_update(X, Q, beta)
            e1.append(np.mean(np.linalg.norm(one - X[idx], axis=1)))
        rows.append({"beta": beta, "retrieval": float(np.mean(acc)), "effective_patterns": float(np.mean(eff)),
                     "steps": float(np.mean(steps)), "one_step_error": float(np.mean(e1))})
    return rows


def glyph_study(noise: float = 0.1, trials: int = 40, seed: int = 0, rules=("hebbian", "pseudoinverse", "storkey", "dam-poly3", "dam-exp")):
    """Recall the ten digits from (a) 10% flips, (b) the bottom 3 rows blanked. Average exact-recovery rate."""
    X = digit_patterns()
    out = []
    for rule in rules:
        rng = np.random.default_rng(seed)
        ok_flip, ok_mask = [], []
        for _ in range(trials):
            Q = flip_noise(X, noise, rng)
            ok_flip.append(np.all(_recall(rule, X, Q, rng, beta=12.0) == X, axis=1))
        idx = np.arange((ROWS - 3) * COLS, ROWS * COLS)
        Qm = mask_noise(X, idx, -1.0)
        o = _recall(rule, X, Qm, rng)
        ok_mask = np.all(o == X, axis=1)
        out.append({"rule": rule, "flip10": float(np.mean(ok_flip)), "bottom3_blank": float(np.mean(ok_mask)),
                    "stored_stable": float(np.mean(_recall(rule, X, X, rng) == X))})
    return out


def glyph_example(rule: str = "dam-exp", digit: int = 3, seed: int = 1):
    """ASCII before/after for one occluded digit."""
    X = digit_patterns()
    rng = np.random.default_rng(seed)
    idx = np.arange((ROWS - 3) * COLS, ROWS * COLS)
    q = mask_noise(X[digit:digit + 1], idx, -1.0)
    out = _recall(rule, X, q, rng)
    a, b = render(q[0]).split("\n"), render(out[0]).split("\n")
    return "\n".join(f"{x}   ->   {y}" for x, y in zip(a, b))


def fmt_table(rows) -> str:
    cols = list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join((f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |\n"
    return head + body
