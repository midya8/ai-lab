"""Synthetic sequence tasks that separate input-dependent dynamics from fixed ones.

Token ids: 0 is noise/padding, 1..V are data symbols, V+1 is the 'recall' marker.
Each generator returns ``(tokens, targets)``; ``targets == -1`` marks positions that are not scored.
"""
from __future__ import annotations

import numpy as np


def selective_copy(n, seq_len, n_data, vocab, rng, fixed_slots=False):
    """Data symbols sit at random positions among noise; after ``seq_len`` tokens the model
    must emit them in order (``n_data`` marker tokens prompt the outputs).

    A fixed convolution can only copy if the symbols sit at fixed offsets (``fixed_slots=True``);
    with random positions it has to *skip* the noise, which needs content-dependent gating.
    """
    T = seq_len + n_data
    tokens = np.zeros((n, T), dtype=np.int64)
    targets = -np.ones((n, T), dtype=np.int64)
    marker = vocab + 1
    for i in range(n):
        if fixed_slots:
            pos = np.linspace(0, seq_len - 1, n_data).astype(int)
        else:
            pos = np.sort(rng.choice(seq_len, n_data, replace=False))
        syms = rng.integers(1, vocab + 1, n_data)
        tokens[i, pos] = syms
        tokens[i, seq_len:] = marker
        targets[i, seq_len:] = syms
    return tokens, targets


def associative_recall(n, n_pairs, n_keys, vocab, rng):
    """k1 v1 k2 v2 ... kn vn  <query key>  ->  its value.

    Keys are 1..n_keys, values n_keys+1..n_keys+vocab (distinct keys per sequence).
    The query (a key already shown) is the second-to-last token and the answer is scored on the last.
    """
    assert n_pairs <= n_keys
    T = 2 * n_pairs + 2
    tokens = np.zeros((n, T), dtype=np.int64)
    targets = -np.ones((n, T), dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_keys, n_pairs, replace=False) + 1
        vals = rng.integers(0, vocab, n_pairs) + n_keys + 1
        tokens[i, 0:2 * n_pairs:2] = keys
        tokens[i, 1:2 * n_pairs:2] = vals
        q = rng.integers(0, n_pairs)
        tokens[i, 2 * n_pairs] = keys[q]
        tokens[i, 2 * n_pairs + 1] = vals[q]  # teacher-forced copy of the answer position
        targets[i, 2 * n_pairs] = vals[q]
    return tokens, targets


def score(logits, targets):
    """(token accuracy, exact-sequence accuracy) over scored positions."""
    mask = targets >= 0
    hit = (logits.argmax(-1) == targets) | ~mask
    tok = float(((logits.argmax(-1) == targets) & mask).sum() / max(mask.sum(), 1))
    return tok, float(hit.all(axis=1).mean())
