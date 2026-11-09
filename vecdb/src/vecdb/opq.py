"""Optimised product quantisation (OPQ, non-parametric variant of Ge et al., 2013).

Plain PQ cuts the vector into contiguous blocks, which is wasteful when variance or
correlation is spread unevenly across coordinates. OPQ learns an orthogonal rotation R so
that the *rotated* data quantises with lower error. Because R is orthogonal, L2 distances
are preserved: ``|q - x| == |Rq - Rx|``, so search simply rotates the query and uses ADC.
"""
from __future__ import annotations

import numpy as np

from .pq import ProductQuantizer


class OPQ:
    def __init__(self, dim: int, m: int, nbits: int = 8, seed: int = 0):
        self.pq = ProductQuantizer(dim, m, nbits, seed)
        self.dim, self.m, self.nbits = dim, m, nbits
        self.R = np.eye(dim, dtype=np.float32)
        self.history: list[float] = []  # reconstruction MSE (original space) after each round

    @property
    def trained(self) -> bool:
        return self.pq.trained

    @property
    def codebooks(self) -> np.ndarray:
        return self.pq.codebooks

    @codebooks.setter
    def codebooks(self, value: np.ndarray) -> None:
        self.pq.codebooks = value

    def rotate(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(X, dtype=np.float32) @ self.R.T)

    def fit(self, X: np.ndarray, rounds: int = 8, iters: int = 10) -> "OPQ":
        """Alternate: (1) fit PQ on rotated data; (2) solve orthogonal Procrustes
        ``min_R |XR^T - decode(encode(XR^T))|`` in closed form via SVD. Round 0 is plain PQ,
        and the best rotation seen (by training MSE) is kept so fitting never does worse
        than PQ on the training set."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        R = np.eye(self.dim, dtype=np.float32)
        best = (np.inf, R, None)
        self.history = []
        for _ in range(rounds + 1):
            Y = X @ R.T
            self.pq.fit(Y, iters=iters)
            Yhat = self.pq.decode(self.pq.encode(Y))
            err = float(np.mean(np.sum((Y - Yhat) ** 2, axis=1)))  # rotation-invariant
            self.history.append(err)
            if err < best[0]:
                best = (err, R.copy(), self.pq.codebooks.copy())
            # R_new = argmin_R |X R^T - Yhat|: SVD of X^T Yhat = U S V^T -> R^T = U V^T
            U, _, Vt = np.linalg.svd(X.T @ Yhat, full_matrices=False)
            R = (U @ Vt).T.astype(np.float32)
        _, self.R, self.pq.codebooks = best
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.pq.encode(self.rotate(X))

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruction back in the original coordinate system."""
        return np.ascontiguousarray(self.pq.decode(codes) @ self.R)

    def adc_table(self, q: np.ndarray) -> np.ndarray:
        return self.pq.adc_table(self.rotate(np.asarray(q, dtype=np.float32).reshape(1, -1))[0])

    def adc_distances(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return self.pq.adc_distances(table, codes)

    def code_bytes(self) -> int:
        return self.pq.code_bytes()

    def mse(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float32)
        return float(np.mean(np.sum((X - self.decode(self.encode(X))) ** 2, axis=1)))
