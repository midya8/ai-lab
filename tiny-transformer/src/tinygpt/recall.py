"""Associative-recall task: a benchmark where the answer exists only in the context.

A sequence is ``k1 v1 ... kn vn q1 a1 ... qm am`` (m = n_queries, default n) with distinct keys drawn from tokens
``[0, n_keys)`` and values from ``[n_keys, n_keys + n_vals)``; each query key repeats one of the
earlier keys and its answer is that key's value. Chance on the final token is ``1 / n_vals``. The first n pairs are random (irreducible
loss), but every query answer is learnable, which gives the model a dense induction signal (a single
query per sequence did not train: accuracy stuck at 1/2 for 2 pairs). Accuracy is scored on the final answer.
"""
from __future__ import annotations

import numpy as np


def vocab_size(n_keys: int, n_vals: int) -> int:
    return n_keys + n_vals


def seq_len(n_pairs: int, n_queries: int = 0) -> int:
    n_queries = n_queries or n_pairs
    return 2 * n_pairs + 2 * n_queries


def make_batch(batch: int, n_pairs: int, n_keys: int, n_vals: int, rng: np.random.Generator, n_queries: int = 0):
    """Return (x, y, answer): x is (B, T-1) inputs, y the shifted targets, answer the true last value."""
    assert n_pairs <= n_keys, "need n_pairs distinct keys"
    n_queries = n_queries or n_pairs
    T = seq_len(n_pairs, n_queries)
    seqs = np.empty((batch, T), dtype=np.int64)
    answer = np.empty(batch, dtype=np.int64)
    for b in range(batch):
        keys = rng.choice(n_keys, size=n_pairs, replace=False)
        vals = n_keys + rng.integers(0, n_vals, size=n_pairs)
        seqs[b, 0:2 * n_pairs:2] = keys
        seqs[b, 1:2 * n_pairs:2] = vals
        qs = rng.integers(0, n_pairs, size=n_queries)
        seqs[b, 2 * n_pairs::2] = keys[qs]
        seqs[b, 2 * n_pairs + 1::2] = vals[qs]
        answer[b] = vals[qs[-1]]
    return seqs[:, :-1], seqs[:, 1:], answer


def recall_accuracy(model, n_pairs: int, n_keys: int, n_vals: int, n: int = 256, seed: int = 123) -> float:
    """Fraction of queries where the argmax at the final position is the stored value."""
    rng = np.random.default_rng(seed)
    x, _, ans = make_batch(n, n_pairs, n_keys, n_vals, rng)
    logits, _, _ = model.forward(x)
    return float((logits[:, -1].argmax(-1) == ans).mean())
