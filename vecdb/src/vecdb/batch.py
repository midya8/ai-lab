"""Multi-threaded batch search: shard the query matrix and search the shards concurrently.

Only worthwhile where the work releases the GIL (BLAS matmuls in flat/IVF/SQ8 scans); the
pure-Python HNSW walk is GIL-bound, so threads give it no speed-up (results stay identical).
Searching is read-only, but each index overwrites ``last_stats`` on every call, so after a
parallel search ``last_stats`` reflects whichever shard finished last and should be ignored.
Do not mutate the index (add/remove/repair) while a parallel search runs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .util import as_matrix


def parallel_search(index, Q, k: int = 10, workers: int = 4, shard: int = 64, **kwargs):
    """Same result as ``index.search(Q, k, **kwargs)``, computed on ``workers`` threads.

    Queries are cut into shards of ``shard`` rows; row order is preserved exactly.
    """
    if workers < 1 or shard < 1:
        raise ValueError("workers and shard must be >= 1")
    Qm = as_matrix(Q, index.dim)
    if workers == 1 or len(Qm) <= shard:
        return index.search(Qm, k, **kwargs)
    parts = [Qm[s:s + shard] for s in range(0, len(Qm), shard)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda q: index.search(q, k, **kwargs), parts))
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])
