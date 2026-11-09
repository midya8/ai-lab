"""PQ vs OPQ: reconstruction MSE and top-10 recall (ADC scan, no re-rank) on synthetic data."""
import numpy as np

from vecdb import FlatIndex, clustered, queries_like, recall_at_k
from vecdb.opq import OPQ
from vecdb.pq import ProductQuantizer


def adc_topk(q, X, k):
    codes = q.encode(X)
    out = []
    for v in Xq:
        d = q.adc_distances(q.adc_table(v), codes)
        out.append(np.argsort(d)[:k])
    return np.array(out)


def make(kind, n=8000, d=32, seed=0):
    X, _ = clustered(n, d, n_clusters=40, spread=0.5, seed=seed)
    if kind == "rotated":  # mix coordinates so structure is not axis-aligned
        Q, _ = np.linalg.qr(np.random.default_rng(seed + 5).normal(size=(d, d)))
        X = (X * np.linspace(2.5, 0.3, d)) @ Q.T
    return X.astype(np.float32)


for kind in ("clustered", "rotated"):
    X = make(kind)
    Xq = queries_like(X, 100)
    truth = np.array([np.argsort(((X - v) ** 2).sum(1))[:10] for v in Xq])
    for name, q in (("PQ ", ProductQuantizer(32, 8, 6)), ("OPQ", OPQ(32, 8, 6))):
        q.fit(X, **({"iters": 10} if name == "PQ " else {"rounds": 8, "iters": 10}))
        r = recall_at_k(adc_topk(q, X, 10), truth)
        print(f"{kind:10s} {name} mse={q.mse(X):8.3f} recall@10={r:.3f}")
