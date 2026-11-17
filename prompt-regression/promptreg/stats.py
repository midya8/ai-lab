"""Small, dependency-free statistics for comparing two prompts on the same cases."""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; well-behaved at 0/1 and small n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    ``b`` = cases only A passed, ``c`` = cases only B passed. Under the null
    each discordant case is a fair coin flip, so this is a binomial sign test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 4000, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (b - a) with a percentile bootstrap CI, resampling cases jointly."""
    if len(a) != len(b):
        raise ValueError("paired samples must be the same length")
    if not a:
        return (0.0, 0.0, 0.0)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (sum(diffs) / n, lo, hi)


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters; ``None`` when chance agreement is 1 (undefined)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        raise ValueError("no items")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def _ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (Pearson on average ranks); ``None`` if either side is constant."""
    if len(x) != len(y):
        raise ValueError("samples must be the same length")
    if len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)
