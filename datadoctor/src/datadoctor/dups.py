"""Exact / near duplicates, conflicting duplicates and train-test leakage."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np


def exact_duplicate_groups(X: np.ndarray) -> list[list[int]]:
    """Groups (size >= 2) of rows with byte-identical feature vectors."""
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    seen: dict[bytes, list[int]] = defaultdict(list)
    for i in range(len(X)):
        seen[X[i].tobytes()].append(i)
    return [g for g in seen.values() if len(g) > 1]


def conflicting_groups(groups: list[list[int]], y: np.ndarray) -> list[list[int]]:
    """Duplicate groups whose members carry more than one distinct label."""
    y = np.asarray(y)
    return [g for g in groups if len(set(int(y[i]) for i in g)) > 1]


def _standardise(X: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = X if ref is None else ref
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def nearest_neighbor_distances(X: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distance from each (standardised) row to its nearest *other* row."""
    Z = _standardise(np.asarray(X, dtype=float))
    out = np.empty(len(Z))
    for s in range(0, len(Z), chunk):
        D = _sqdist(Z[s : s + chunk], Z)
        D[np.arange(D.shape[0]), np.arange(s, s + D.shape[0])] = np.inf
        out[s : s + chunk] = np.sqrt(D.min(axis=1))
    return out


def default_eps(X: np.ndarray, fraction: float = 0.05) -> float:
    """Near-duplicate radius: a small fraction of the typical nearest-neighbour distance."""
    return float(fraction * np.median(nearest_neighbor_distances(X)))


def near_duplicate_pairs(X: np.ndarray, eps: float | None = None, chunk: int = 512) -> list[tuple[int, int, float]]:
    """Pairs (i<j, distance) closer than ``eps`` in standardised feature space."""
    X = np.asarray(X, dtype=float)
    eps = default_eps(X) if eps is None else eps
    Z = _standardise(X)
    pairs = []
    for s in range(0, len(Z), chunk):
        D = _sqdist(Z[s : s + chunk], Z)
        r, c = np.nonzero(D <= eps * eps)
        for a, b in zip(r, c):
            i = s + int(a)
            if i < b:
                pairs.append((i, int(b), float(np.sqrt(D[a, b]))))
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    return pairs


def leakage_pairs(
    train_X: np.ndarray, test_X: np.ndarray, eps: float | None = None, chunk: int = 512
) -> list[tuple[int, int, float]]:
    """(train_idx, test_idx, distance) for test rows within ``eps`` of a training row.

    Both sets are standardised with the *training* statistics, as a model would see them.
    """
    train_X = np.asarray(train_X, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    eps = default_eps(train_X) if eps is None else eps
    A = _standardise(train_X)
    B = _standardise(test_X, ref=train_X)
    out = []
    for s in range(0, len(B), chunk):
        D = _sqdist(B[s : s + chunk], A)
        r, c = np.nonzero(D <= eps * eps)
        for a, b in zip(r, c):
            out.append((int(b), s + int(a), float(np.sqrt(D[a, b]))))
    out.sort(key=lambda p: (p[2], p[0], p[1]))
    return out


# ---- text: MinHash + LSH ----------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, n: int = 3) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _h64(s: str, salt: int) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8, salt=salt.to_bytes(8, "little")).digest(), "little")


def minhash_signature(sh: set[str], num_perm: int = 64) -> tuple[int, ...]:
    if not sh:
        return tuple([0] * num_perm)
    return tuple(min(_h64(s, p) for s in sh) for p in range(num_perm))


def near_duplicate_texts(texts: list[str], threshold: float = 0.8, num_perm: int = 64, bands: int = 16, n: int = 3):
    """Pairs (i<j, exact Jaccard) of texts with shingle-Jaccard >= threshold.

    MinHash signatures are bucketed into LSH bands to find candidates cheaply; every
    candidate is then verified with the exact Jaccard, so there are no false positives
    (misses are possible only for pairs near the threshold).
    """
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    sets = [shingles(t, n) for t in texts]
    sigs = [minhash_signature(s, num_perm) for s in sets]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        if not sets[i]:
            continue
        for b in range(bands):
            buckets[(b, sig[b * rows : (b + 1) * rows])].append(i)
    cand: set[tuple[int, int]] = set()
    for members in buckets.values():
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                cand.add((members[a], members[c]))
    out = []
    for i, j in cand:
        jac = jaccard(sets[i], sets[j])
        if jac >= threshold:
            out.append((i, j, jac))
    out.sort(key=lambda p: (-p[2], p[0], p[1]))
    return out
