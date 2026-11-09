"""Synthetic datasets with known defects, so detectors can be scored against ground truth."""

from __future__ import annotations

import numpy as np


def make_blobs(n: int = 1500, n_classes: int = 5, dim: int = 12, sep: float = 0.9, seed: int = 0):
    """Gaussian classes with overlapping tails (``sep`` is the centre spread in std units)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(n_classes, dim))
    scales = rng.uniform(0.8, 1.4, size=n_classes)
    y = np.arange(n) % n_classes
    rng.shuffle(y)
    X = centers[y] + rng.normal(size=(n, dim)) * scales[y][:, None]
    return X, y


def inject_label_noise(y: np.ndarray, rate: float, n_classes: int | None = None, kind: str = "symmetric", seed: int = 0):
    """Flip ``rate`` of the labels. Returns (noisy_labels, flipped_mask).

    symmetric   new label uniform over the other classes
    pairwise    class c always becomes (c + 1) mod K  (systematic confusion)
    asymmetric  each class has its own favourite wrong label (random permutation without fixed points),
                chosen 80% of the time, otherwise uniform
    """
    y = np.asarray(y, dtype=int)
    K = int(n_classes if n_classes is not None else y.max() + 1)
    rng = np.random.default_rng(seed)
    n = len(y)
    m = int(round(rate * n))
    flip = np.zeros(n, dtype=bool)
    flip[rng.choice(n, size=m, replace=False)] = True
    new = y.copy()
    fav = (np.arange(K) + 1) % K
    if kind == "asymmetric":
        perm = rng.permutation(K)
        while (perm == np.arange(K)).any():
            perm = rng.permutation(K)
        fav = perm
    for i in np.flatnonzero(flip):
        if kind == "pairwise":
            new[i] = fav[y[i]]
        elif kind == "asymmetric" and rng.random() < 0.8:
            new[i] = fav[y[i]]
        else:
            others = [c for c in range(K) if c != y[i]]
            new[i] = others[rng.integers(len(others))]
    return new, flip


def add_duplicates(X: np.ndarray, y: np.ndarray, frac: float = 0.05, jitter: float = 0.0, seed: int = 0):
    """Append copies of random rows (optionally jittered). Returns (X2, y2, source_index_of_each_copy)."""
    rng = np.random.default_rng(seed)
    m = int(round(frac * len(X)))
    src = rng.choice(len(X), size=m, replace=False)
    Xc = X[src] + (rng.normal(size=(m, X.shape[1])) * jitter if jitter else 0.0)
    return np.vstack([X, Xc]), np.concatenate([y, y[src]]), src


def add_outliers(X: np.ndarray, y: np.ndarray, count: int = 10, scale: float = 8.0, seed: int = 0):
    """Append far-away points carrying real class labels. Returns (X2, y2, outlier_mask)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    direction = rng.normal(size=(count, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    centre = X.mean(0)
    far = centre + direction * scale * X.std(0).mean() * np.sqrt(d) / 2
    ylab = rng.integers(0, int(y.max()) + 1, size=count)
    mask = np.zeros(len(X) + count, dtype=bool)
    mask[len(X) :] = True
    return np.vstack([X, far]), np.concatenate([y, ylab]), mask


def make_modes(n: int = 1500, n_classes: int = 4, dim: int = 8, sep: float = 2.5, seed: int = 0):
    """Each class is two far-apart Gaussian modes placed so that classes are XOR-like:
    a linear model cannot separate them, a local method can."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(n_classes * 2, dim))
    owner = np.repeat(np.arange(n_classes), 2)
    y = np.arange(n) % n_classes
    rng.shuffle(y)
    mode = rng.integers(0, 2, size=n)
    X = centers[y * 2 + mode] + rng.normal(size=(n, dim))
    return X, y


def inject_instance_noise(X: np.ndarray, y: np.ndarray, rate: float, temperature: float = 0.5, seed: int = 0):
    """Instance-dependent noise: errors concentrate near class borders.

    A row's flip weight is softmax(-margin / temperature) where margin is (distance to own class centroid
    minus distance to the nearest other centroid), standardised; rows are sampled without replacement
    for exactly round(rate * n) flips, and each flipped row takes the label of its nearest other centroid
    (the plausible human mistake). Returns (noisy_labels, flipped_mask)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    K = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    cent = np.stack([X[y == c].mean(0) for c in range(K)])
    d = np.linalg.norm(X[:, None, :] - cent[None], axis=2)
    own = d[np.arange(len(y)), y]
    other = d.copy()
    other[np.arange(len(y)), y] = np.inf
    nearest = other.argmin(1)
    gap = own - other[np.arange(len(y)), nearest]  # larger = closer to the border
    z = (gap - gap.mean()) / (gap.std() + 1e-12)
    w = np.exp((z - z.max()) / temperature)
    w /= w.sum()
    m = int(round(rate * len(y)))
    flip = np.zeros(len(y), dtype=bool)
    flip[rng.choice(len(y), size=m, replace=False, p=w)] = True
    new = y.copy()
    new[flip] = nearest[flip]
    return new, flip
