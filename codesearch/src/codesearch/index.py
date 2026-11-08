"""Field-weighted BM25 (BM25F) and LSA over code chunks."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .chunker import Chunk
from .tokens import terms

FIELDS = ("name", "owner", "doc", "sig", "body", "path")
DEFAULT_WEIGHTS = {"name": 3.0, "owner": 1.0, "doc": 2.0, "sig": 1.0, "body": 1.0, "path": 0.8}
DEFAULT_B = {"name": 0.3, "owner": 0.3, "doc": 0.75, "sig": 0.5, "body": 0.75, "path": 0.3}


def chunk_fields(c: Chunk) -> dict[str, list[str]]:
    """The terms of each field of a chunk."""
    path_parts = c.path.replace("\\", "/").rsplit(".", 1)[0]
    sig_text = c.signature if c.kind != "module" else ""
    return {
        "name": terms(c.name.replace("<module>", "")),
        "owner": terms(c.qualname.rsplit(".", 1)[0]) if "." in c.qualname else [],
        "doc": terms(c.doc),
        "sig": terms(sig_text),
        "body": terms(" ".join(c.ids)) + terms(" ".join(c.strings)) + terms(" ".join(c.comments)),
        "path": terms(path_parts.replace("/", " ")),
    }


@dataclass
class Index:
    """Immutable-ish search index. Build with ``Index.build(chunks)``."""

    chunks: list[Chunk]
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    b: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_B))
    k1: float = 1.2
    lsa_dim: int = 96
    seed: int = 0

    def __post_init__(self) -> None:
        self.n = len(self.chunks)
        self.fields = [chunk_fields(c) for c in self.chunks]
        self.id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
        self._build_bm25f()
        self._build_flat()
        self._lsa = None

    @classmethod
    def build(cls, chunks: list[Chunk], **kw) -> "Index":
        return cls(list(chunks), **kw)

    # ------------------------------------------------------------------ BM25F
    def _build_bm25f(self) -> None:
        avg = {}
        for f in FIELDS:
            lens = [len(d[f]) for d in self.fields]
            avg[f] = (sum(lens) / len(lens)) if lens and sum(lens) else 1.0
        post: dict[str, dict[int, float]] = {}
        df: Counter = Counter()
        for i, d in enumerate(self.fields):
            tf: dict[str, float] = {}
            for f in FIELDS:
                toks = d[f]
                if not toks:
                    continue
                norm = 1 - self.b[f] + self.b[f] * len(toks) / avg[f]
                for t, c in Counter(toks).items():
                    tf[t] = tf.get(t, 0.0) + self.weights[f] * c / norm
            for t, v in tf.items():
                post.setdefault(t, {})[i] = v
                df[t] += 1
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings = {
            t: (np.fromiter(m.keys(), dtype=np.int64, count=len(m)),
                np.fromiter(m.values(), dtype=np.float64, count=len(m)))
            for t, m in post.items()
        }
        self._tf = post

    def bm25f(self, query_terms: list[str]) -> np.ndarray:
        scores = np.zeros(self.n)
        for t, qtf in Counter(query_terms).items():
            if t not in self.postings:
                continue
            idx, v = self.postings[t]
            scores[idx] += self.idf[t] * (1 + math.log(qtf)) * v / (self.k1 + v)
        return scores

    # -------------------------------------------------- flat BM25 (ablation baseline)
    def _build_flat(self) -> None:
        docs = [sum((d[f] for f in FIELDS), []) for d in self.fields]
        self._flat_len = np.array([len(d) for d in docs], dtype=float)
        avg = self._flat_len.mean() if self.n and self._flat_len.sum() else 1.0
        self._flat_avg = avg
        post: dict[str, dict[int, int]] = {}
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                post.setdefault(t, {})[i] = c
        self._flat_post = {
            t: (np.fromiter(m.keys(), dtype=np.int64), np.fromiter(m.values(), dtype=float))
            for t, m in post.items()
        }
        self._flat_idf = {t: math.log(1 + (self.n - len(m) + 0.5) / (len(m) + 0.5)) for t, m in post.items()}

    def bm25_flat(self, query_terms: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(self.n)
        for t in set(query_terms):
            if t not in self._flat_post:
                continue
            idx, tf = self._flat_post[t]
            norm = k1 * (1 - b + b * self._flat_len[idx] / self._flat_avg)
            scores[idx] += self._flat_idf[t] * tf * (k1 + 1) / (tf + norm)
        return scores

    def name_only(self, query_terms: list[str]) -> np.ndarray:
        """grep-like baseline: number of query terms found in the symbol name."""
        qs = set(query_terms)
        return np.array([len(qs & set(d["name"])) for d in self.fields], dtype=float)

    # -------------------------------------------------------------------- LSA
    def _ensure_lsa(self) -> dict | None:
        if self._lsa is not None:
            return self._lsa or None
        vocab_terms = [t for t, m in self._tf.items() if len(m) >= 2]
        if self.n < 3 or len(vocab_terms) < 3:
            self._lsa = {}
            return None
        vocab_terms.sort(key=lambda t: -len(self._tf[t]))
        vocab_terms = vocab_terms[:20000]
        vocab = {t: j for j, t in enumerate(vocab_terms)}
        X = np.zeros((self.n, len(vocab)), dtype=np.float32)
        for t, j in vocab.items():
            for i, v in self._tf[t].items():
                X[i, j] = math.log1p(v) * self.idf[t]
        k = max(2, min(self.lsa_dim, self.n - 1, len(vocab) - 1))
        Vk = _randomized_right_vectors(X, k, self.seed)
        D = X @ Vk
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        D = D / np.where(norms == 0, 1, norms)
        self._lsa = {"vocab": vocab, "Vk": Vk, "D": D}
        return self._lsa

    def _query_vec(self, query_terms: list[str], lsa: dict) -> np.ndarray | None:
        q = np.zeros(len(lsa["vocab"]), dtype=np.float32)
        for t, c in Counter(query_terms).items():
            j = lsa["vocab"].get(t)
            if j is not None:
                q[j] = math.log1p(c) * self.idf[t]
        z = q @ lsa["Vk"]
        nz = np.linalg.norm(z)
        return None if nz == 0 else z / nz

    def lsa(self, query_terms: list[str]) -> np.ndarray:
        lsa = self._ensure_lsa()
        if not lsa:
            return np.zeros(self.n)
        z = self._query_vec(query_terms, lsa)
        if z is None:
            return np.zeros(self.n)
        return np.clip(lsa["D"] @ z, 0, None).astype(np.float64)

    def lsa_neighbors(self, idx: int) -> np.ndarray:
        """Cosine similarity of chunk ``idx`` to every chunk in LSA space."""
        lsa = self._ensure_lsa()
        if not lsa:
            return np.zeros(self.n)
        return (lsa["D"] @ lsa["D"][idx]).astype(np.float64)

    def lsa_matrix(self) -> np.ndarray | None:
        lsa = self._ensure_lsa()
        return None if not lsa else lsa["D"]

    def chunk_terms(self, idx: int) -> list[str]:
        """All terms of a chunk (for query-by-example)."""
        d = self.fields[idx]
        return sum((d[f] for f in ("name", "doc", "sig", "body")), [])


def _randomized_right_vectors(X: np.ndarray, k: int, seed: int, oversample: int = 10, power: int = 3) -> np.ndarray:
    """Top-k right singular vectors of X by randomized range finding (Halko et al.)."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((X.shape[1], min(k + oversample, min(X.shape)))).astype(np.float32)
    Y = X @ omega
    for _ in range(power):
        Y, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(X.T @ Y)
        Y = X @ Z
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return np.ascontiguousarray(Vt[:k].T)
