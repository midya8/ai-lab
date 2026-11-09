"""Product quantisation with asymmetric distance computation (ADC)."""
from __future__ import annotations

import numpy as np

from .util import kmeans, sqdist


class ProductQuantizer:
    """Split a vector into ``m`` sub-vectors and quantise each with its own 2**nbits codebook.

    A d-dimensional float32 vector (4d bytes) becomes ``m`` bytes. Queries stay
    uncompressed (asymmetric): ``adc_table`` precomputes the squared distance from each
    query sub-vector to every centroid, and a stored code's distance is then m table lookups.
    """

    def __init__(self, dim: int, m: int, nbits: int = 8, seed: int = 0):
        if m < 1 or dim % m:
            raise ValueError(f"dim={dim} must be divisible by m={m}")
        if not 1 <= nbits <= 8:
            raise ValueError("nbits must be in 1..8 (codes are stored as uint8)")
        self.dim, self.m, self.nbits, self.seed = dim, m, nbits, seed
        self.dsub = dim // m
        self.ksub = 1 << nbits
        self.codebooks: np.ndarray | None = None  # (m, ksub, dsub)

    @property
    def trained(self) -> bool:
        return self.codebooks is not None

    def fit(self, X: np.ndarray, iters: int = 20) -> "ProductQuantizer":
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {X.shape[1]}")
        if len(X) < self.ksub:
            raise ValueError(f"need at least {self.ksub} training vectors for nbits={self.nbits}, got {len(X)}")
        books = np.empty((self.m, self.ksub, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = X[:, j * self.dsub:(j + 1) * self.dsub]
            books[j], _, _ = kmeans(sub, self.ksub, iters=iters, seed=self.seed + j)
        self.codebooks = books
        return self

    def _need(self):
        if self.codebooks is None:
            raise RuntimeError("ProductQuantizer is not trained; call fit() first")

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._need()
        X = np.ascontiguousarray(X, dtype=np.float32)
        codes = np.empty((len(X), self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = X[:, j * self.dsub:(j + 1) * self.dsub]
            codes[:, j] = sqdist(sub, self.codebooks[j]).argmin(axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        self._need()
        codes = np.asarray(codes)
        out = np.empty((len(codes), self.dim), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self.dsub:(j + 1) * self.dsub] = self.codebooks[j][codes[:, j]]
        return out

    def adc_table(self, q: np.ndarray) -> np.ndarray:
        """(m, ksub) squared distances from the query's sub-vectors to every centroid."""
        self._need()
        q = np.asarray(q, dtype=np.float32).reshape(self.m, 1, self.dsub)
        diff = self.codebooks - q
        return np.einsum("mkd,mkd->mk", diff, diff)

    def adc_distances(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Squared distance from the query to each decoded code, by table lookup."""
        return table[np.arange(self.m)[None, :], codes].sum(axis=1)

    def code_bytes(self) -> int:
        return self.m

    def mse(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error per vector."""
        R = self.decode(self.encode(X))
        return float(np.mean(np.sum((np.asarray(X, dtype=np.float32) - R) ** 2, axis=1)))
