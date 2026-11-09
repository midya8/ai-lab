"""Hybrid retrieval for docqa: ragkit's BM25 fused with a char-n-gram dense index from vecdb.

The dense side is a hashed character-trigram embedding (no model download). It is
robust to typos and morphology, which is where pure BM25 is weakest. Both rankings
are fused with reciprocal-rank fusion by ``vecdb.HybridCollection``.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

try:
    import vecdb  # noqa: F401
except ImportError:  # pragma: no cover - monorepo fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vecdb" / "src"))

from vecdb.hybrid import HybridCollection  # noqa: E402
from ragkit.retrieve import Retriever, ScoredChunk  # noqa: E402


def embed_ngrams(text: str, dim: int = 512, n: int = 3) -> np.ndarray:
    """Signed feature-hashed character n-grams, sublinear tf, L2-normalised."""
    s = f" {' '.join(text.lower().split())} "
    v = np.zeros(dim, dtype=np.float32)
    for i in range(max(len(s) - n + 1, 0)):
        h = zlib.crc32(s[i:i + n].encode())
        v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    v = np.sign(v) * np.sqrt(np.abs(v))
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


class HybridRetriever:
    """Drop-in for ragkit's ``Retriever`` in DocQA: same ``chunks``, ``_df`` and ``search``."""

    def __init__(self, chunks, dim: int = 512, fusion: str = "rrf"):
        self.base = Retriever(list(chunks))
        self.chunks = self.base.chunks
        self._df = self.base._df
        self.dim = dim
        self.fusion = fusion
        self.col = HybridCollection(dim)
        if self.chunks:
            ids = [str(i) for i in range(len(self.chunks))]
            texts = [c.text for c in self.chunks]
            self.col.upsert(ids, np.stack([embed_ngrams(t, dim) for t in texts]), texts=texts)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        hits = self.col.query(embed_ngrams(query, self.dim), query, k=k, fusion=self.fusion)
        return [ScoredChunk(self.chunks[int(h.id)], -h.distance) for h in hits]
