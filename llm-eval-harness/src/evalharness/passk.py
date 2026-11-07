"""Unbiased pass@k estimator (Chen et al., 2021) and sampling-based evaluation."""
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples drawn from n (c correct) passes."""
    if n < 1 or k < 1 or k > n or not 0 <= c <= n:
        raise ValueError("require 1 <= k <= n and 0 <= c <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def evaluate_pass_at_k(backend, cases, scorer_fn, n=5, ks=(1, 5), threshold=1.0):
    """Sample each prompt n times; return mean pass@k per k. cases: [(prompt, case)]."""
    ks = [k for k in ks if k <= n]
    totals = {k: 0.0 for k in ks}
    for prompt, case in cases:
        c = sum(1 for _ in range(n) if scorer_fn(backend.generate(prompt), case) >= threshold)
        for k in ks:
            totals[k] += pass_at_k(n, c, k)
    m = max(len(cases), 1)
    return {f"pass@{k}": v / m for k, v in totals.items()}
