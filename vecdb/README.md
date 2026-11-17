# vecdb

Nearest-neighbour search and a small vector database in NumPy. Indexes: exact flat, IVF-Flat, IVF-PQ (with optional OPQ and exact re-ranking), SQ8, and HNSW written from the paper. On top sits a `Collection` with string ids, JSON metadata, Mongo-style filters, upsert/delete and on-disk persistence (no pickle), plus a hybrid BM25 + vector collection.

```bash
cd vecdb
pip install -r requirements.txt
PYTHONPATH=src python -m vecdb demo
PYTHONPATH=src python -m vecdb bench --n 5000 --dim 64
PYTHONPATH=src python experiments/benchmark.py        # full tables, ~7 min
PYTHONPATH=src python -m pytest -q
```

```python
from vecdb import Collection

c = Collection(dim=384, metric="cosine", index="hnsw", M=16, ef_construction=100)
c.upsert(["a", "b"], vectors, [{"lang": "en", "year": 2024}, {"lang": "fr", "year": 2021}])
hits = c.query(query_vector, k=5, where={"lang": "en", "year": {"$gte": 2023}}, ef=64)
print(c.tune(k=10, target_recall=0.95))    # cheapest ef/nprobe that reaches 95% recall
c.save("my_collection")
```

## What the benchmarks showed

20k synthetic vectors, 64 dims, k=10, one run on one machine:

| data | method | recall@10 | distance evals | bytes/vector |
|---|---|---|---|---|
| clustered | exact | 1.000 | 20,000 | 256 |
| clustered | IVF-Flat, nprobe 2 | 0.987 | 184 | 266 |
| clustered | IVF-PQ, ADC only | 0.744 | 1,179 | 29 |
| clustered | IVF-PQ + re-rank | 0.994 | 1,179 | 285 |
| clustered | HNSW, ef 10 | 0.992 | 236 | 360 |
| gaussian | IVF-Flat, nprobe 32 | 0.711 | 4,761 | 266 |
| gaussian | HNSW, ef 160 | 0.971 | 3,278 | 354 |

- Data structure decides almost everything. IVF is brilliant on clusters and poor on unstructured data, where HNSW holds up better.
- HNSW does 20-80x fewer distance evaluations but isn't faster here. At this size one BLAS pass is so cheap that Python graph traversal loses on wall-clock.
- PQ buys memory (9x smaller), not accuracy. Recall plateaus at 0.744 from quantisation error, and getting it back means keeping the raw vectors.
- With selective filters, don't walk the graph. At 1% selectivity scanning the matching rows needed 199 evaluations vs 13,430 for the walk, so the index falls back to brute force below 15% (picked from this one experiment).
- Hybrid fusion didn't beat the best single retriever on my synthetic mixed queries. BM25 alone wins lookups.
- OPQ helps a bit, and only when variance isn't axis-aligned.

All synthetic data. Real embeddings, more than 20k vectors and concurrent writers are untested.
