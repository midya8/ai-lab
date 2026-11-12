"""Zipf fit: how closely emitted-token frequencies follow a power law."""
from __future__ import annotations
import math
from collections import Counter


def zipf_fit(tok, text: str, top: int | None = None):
    """Least-squares fit of log(freq) against log(rank) over emitted tokens.

    Returns tokens, distinct, slope (Zipf's law predicts about -1), intercept
    and r2 (fit quality, 1.0 = perfect power law). `top` restricts the fit to
    the most frequent tokens. Needs at least 2 distinct tokens, otherwise
    slope/r2 are None."""
    c = Counter(tok.encode(text))
    freqs = sorted(c.values(), reverse=True)
    if top is not None:
        freqs = freqs[:top]
    out = {"tokens": sum(c.values()), "distinct": len(c),
           "slope": None, "intercept": None, "r2": None}
    if len(freqs) < 2:
        return out
    xs = [math.log(i + 1) for i in range(len(freqs))]
    ys = [math.log(f) for f in freqs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    out["slope"], out["intercept"] = slope, my - slope * mx
    out["r2"] = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return out
