"""Index persistence: one ``.npz`` per index (arrays + JSON metadata, no pickle)."""
from __future__ import annotations

import json
import os

import numpy as np

from .flat import FlatIndex
from .hnsw import HNSWIndex
from .ivf import IVFIndex
from .sq8 import SQ8Index

_KINDS = {"flat": FlatIndex, "ivf": IVFIndex, "hnsw": HNSWIndex, "sq8": SQ8Index}
FORMAT = 1


def save_index(index, path: str) -> None:
    arrays, meta = index._state()
    meta = dict(meta, kind=index.kind, format=FORMAT)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    os.replace(tmp, path)


def load_index(path: str):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
    if meta.get("format") != FORMAT:
        raise ValueError(f"unsupported index format {meta.get('format')!r}")
    cls = _KINDS.get(meta["kind"])
    if cls is None:
        raise ValueError(f"unknown index kind {meta['kind']!r}")
    return cls._from_state(arrays, meta)
