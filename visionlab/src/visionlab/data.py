"""Procedural 'shapes' dataset: six classes, rendered anti-aliased, fully seeded.

Each image is a random shape (random size, position, rotation, brightness) on a
noisy dark background, so the task is small enough for a NumPy CNN yet needs real
translation/rotation invariance. All data is synthetic; accuracy numbers say
nothing about natural images.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["disk", "ring", "square", "triangle", "cross", "stripes"]


def _shape_mask(cls: str, u, v, r):
    """Boolean mask of a shape of 'radius' r in its own (rotated) frame."""
    if cls == "disk":
        return u * u + v * v <= (0.9 * r) ** 2
    if cls == "ring":
        d2 = u * u + v * v
        return (d2 <= r * r) & (d2 >= (0.6 * r) ** 2)
    if cls == "square":
        return np.maximum(np.abs(u), np.abs(v)) <= 0.8 * r
    if cls == "triangle":  # equilateral, circumradius r, inradius r/2
        inside = np.ones_like(u, dtype=bool)
        for k in range(3):
            phi = np.deg2rad(90 + 120 * k)
            inside &= (u * np.cos(phi) + v * np.sin(phi)) <= 0.5 * r
        return inside
    if cls == "cross":
        w = 0.25 * r
        return ((np.abs(u) <= w) & (np.abs(v) <= r)) | ((np.abs(v) <= w) & (np.abs(u) <= r))
    if cls == "stripes":  # three bars out of five equal bands
        band = np.floor((v + 0.9 * r) / (0.36 * r)).astype(int)
        return (np.abs(u) <= 0.9 * r) & (np.abs(v) <= 0.9 * r) & (band % 2 == 0)
    raise ValueError(f"unknown class {cls!r}")


def render(cls: str, rng: np.random.Generator, size: int = 24, noise: float = 0.2, ss: int = 3):
    """Render one image as float32 (size, size) in [0, 1]."""
    n = size * ss
    c = (np.arange(n) + 0.5) / ss
    X, Y = np.meshgrid(c, c)
    r = rng.uniform(0.21, 0.35) * size
    cx = size / 2 + rng.uniform(-0.1, 0.1) * size
    cy = size / 2 + rng.uniform(-0.1, 0.1) * size
    th = rng.uniform(0, 2 * np.pi)
    ct, st = np.cos(th), np.sin(th)
    u = ct * (X - cx) + st * (Y - cy)
    v = -st * (X - cx) + ct * (Y - cy)
    mask = _shape_mask(cls, u, v, r).astype(np.float32)
    mask = mask.reshape(size, ss, size, ss).mean(axis=(1, 3))  # box-filter anti-aliasing
    fg = rng.uniform(0.6, 1.0)
    bg = rng.uniform(0.0, 0.15)
    img = bg + (fg - bg) * mask
    img = img + rng.normal(0, rng.uniform(0, noise), img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def make_dataset(n: int, seed: int = 0, size: int = 24, noise: float = 0.2):
    """Balanced, shuffled dataset: X (n,1,size,size) float32, y (n,) int64."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % len(CLASSES)
    rng.shuffle(y)
    X = np.stack([render(CLASSES[k], rng, size, noise) for k in y])[:, None]
    return X, y.astype(np.int64)


def augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label-preserving flips and 90-degree rotations (every class is invariant to them)."""
    out = np.empty_like(X)
    ks = rng.integers(0, 4, len(X))
    flips = rng.random(len(X)) < 0.5
    for i, (k, f) in enumerate(zip(ks, flips)):
        img = np.rot90(X[i], k, axes=(1, 2))
        out[i] = img[:, :, ::-1] if f else img
    return out


def ascii_art(img: np.ndarray) -> str:
    """Tiny terminal rendering of a (H,W) or (1,H,W) image."""
    img = np.asarray(img).reshape(img.shape[-2:])
    ramp = " .:-=+*#%@"
    idx = np.clip((img * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
    return "\n".join("".join(ramp[i] * 2 for i in row) for row in idx)
