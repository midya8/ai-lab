"""Dependency-free PNG writer and heat-map montage helpers."""
from __future__ import annotations

import struct
import zlib

import numpy as np


def write_png(path, arr):
    """Write a uint8 (H,W) grayscale or (H,W,3) RGB array as a PNG file."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        raise ValueError("write_png needs uint8")
    if a.ndim == 2:
        ctype, rows = 0, a
    elif a.ndim == 3 and a.shape[2] == 3:
        ctype, rows = 2, a.reshape(a.shape[0], -1)
    else:
        raise ValueError("shape must be (H,W) or (H,W,3)")
    raw = b"".join(b"\x00" + rows[i].tobytes() for i in range(rows.shape[0]))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 8, ctype, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def read_png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    w, h = struct.unpack(">II", head[16:24])
    return h, w


def upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def gray_rgb(img01):
    g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def heat_rgb(h01):
    """Black -> red -> yellow -> white colour ramp."""
    h = np.clip(h01, 0, 1)
    r = np.clip(h * 3, 0, 1)
    g = np.clip(h * 3 - 1, 0, 1)
    b = np.clip(h * 3 - 2, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(img01, h01, alpha=0.65):
    base = gray_rgb(img01).astype(float)
    heat = heat_rgb(h01).astype(float)
    a = alpha * np.clip(h01, 0, 1)[..., None]
    return (base * (1 - a) + heat * a).astype(np.uint8)


def montage(rows, pad=2, bg=255):
    """rows: list of lists of equally sized (h,w,3) uint8 tiles -> one image."""
    th, tw = rows[0][0].shape[:2]
    H = len(rows) * (th + pad) + pad
    W = len(rows[0]) * (tw + pad) + pad
    out = np.full((H, W, 3), bg, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            y, x = pad + i * (th + pad), pad + j * (tw + pad)
            out[y:y + th, x:x + tw] = tile
    return out
