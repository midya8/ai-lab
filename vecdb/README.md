# vecdb

Approximate nearest-neighbour search and a small vector database, from scratch in NumPy: an exact **flat** index, **IVF-Flat**, **IVF-PQ** (product-quantised residuals with ADC lookup tables and optional exact re-ranking), and **HNSW** written from the paper, plus a `Collection` layer with string ids, JSON metadata, Mongo-style **filtered search**, upsert/delete, and on-disk persistence (no pickle).

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m vecdb demo                       # filtered k-NN on a small collection
python -m vecdb bench --n 5000 --dim 64    # recall / latency / memory table for every index
python experiments/benchmark.py            # the full tables in experiments/results.md (~7 min)
python -m pytest -q                        # 76 tests, ~10 s
```

```python
from vecdb import Collection

c = Collection(dim=384, metric="cosine", index="hnsw", M=16, ef_construction=100)
c.upsert(["a", "b"], vectors, [{"lang": "en", "year": 2024}, {"lang": "fr", "year": 2021}])
hits = c.query(query_vector, k=5, where={"lang": "en", "year": {"$gte": 2023}}, ef=64)
c.save("my_collection"); c = Collection.load("my_collection")
```

## What is in it

| piece | what it does |
|---|---|
| `FlatIndex` | Exact search (chunked BLAS distance matrix, deterministic tie-breaking). The ground truth for every recall number. |
| `IVFIndex` | k-means (k-means++ seeding, empty-cluster repair) partitions the data; a query scans `nprobe` cells. Vectors added before `train()` are buffered. With `pq_m=` the cells hold `m`-byte PQ codes of the *residual* to the centroid, scored by ADC table lookups; `opq=True` rotates residuals with a learned OPQ matrix before PQ (persisted); on synthetic rotated Gaussian data (n=6000, d=32, m=8, 6 bits, full probe) residual MSE fell 14.04 -> 13.32 and recall@10 rose 0.539 -> 0.567, a modest single-seed gain. `keep_raw=True` enables `refine=r`, an exact re-rank of the top `r*k` ADC hits. |
| `ProductQuantizer` | Per-subspace codebooks, encode/decode, `adc_table`. Tested: ADC distance equals the distance to the decoded vector exactly. |
| `SQ8Index` | Scalar quantisation: one uint8 per dimension (4x less memory than float32), per-dimension range fitted on the first batch (later outliers clip). Asymmetric search (float32 query, decoded chunks); `keep_raw=True` + `refine=r` re-ranks the top `r*k` exactly. Synthetic 20k x 64, batched 200 queries, k=10: recall 0.980 (clusters) / 0.981 (gaussian) without refine, 1.000 with `refine=2`; it decodes in Python so it is not faster than flat float32 (~0.25 ms/query vs ~0.2) - the win is memory, not speed. Persisted by `save_index`/`load_index`. |
| `HNSWIndex` | Layered proximity graph: geometric level draw, greedy descent, `ef_construction` candidate search, the paper's diversity heuristic (Algorithm 4) for neighbour selection and for pruning overfull neighbour lists, tombstone deletes, `repair()` to unlink them (deleted nodes' live neighbours become candidates, re-selected by the diversity heuristic; entry point re-chosen), save/load that can keep inserting. Repair finding (5k x 32 synthetic, 30-80% deleted, 100 queries): recall@10 was already ~1.0 with tombstones and stayed ~1.0 after repair, so repair did not buy recall here; it cut distance evaluations per query by 25-76% (e.g. 858 -> 204 at 80% deleted, ef=60) because searches stop wading through dead nodes. One seed, one dataset. |
| `Collection` | ids, metadata, `where=` filters pushed into the index search, upsert (tombstone + append), delete, `compact()`, atomic save/load. |
| `filters` | `{"year": {"$gte": 2020}}`, `$in/$nin/$ne/$and/$or/$not`; validated at compile time; type mismatches mean "no match" rather than a crash. |

Metrics: `l2` (squared), `cosine` (vectors are normalised on the way in; reported as `1 - cos`), `ip` (negative dot; not available with PQ because residual PQ needs an L2-compatible metric).

## Measured

Synthetic data only (no downloads): 20,000 vectors, 64 dimensions, 200 queries (data points + 10% noise), k=10, ground truth from the exact index. `clusters` = 100 overlapping Gaussian blobs; `gaussian` = isotropic noise, which has no structure to exploit. Everything below is in [`experiments/results.md`](experiments/results.md); the setup was fixed before running. One machine, one run, single-query Python loops: **read latencies as relative, not absolute.**

Recall@10 and cost, selected rows:

| data | method | setting | recall@10 | ms/query | distance evals | bytes/vector |
|---|---|---|---|---|---|---|
| clusters | exact | - | 1.000 | 0.56 | 20,000 | 256 |
| clusters | IVF-Flat | nprobe=2 | 0.987 | 0.09 | 184 | 266 |
| clusters | IVF-PQ m=16 | nprobe=8, ADC only | 0.744 | 0.48 | 1,179 | 29 |
| clusters | IVF-PQ m=16 | nprobe=8, + exact re-rank x4 | 0.994 | 0.52 | 1,179 | 285 |
| clusters | HNSW | ef=10 | 0.992 | 0.32 | 236 | 360 |
| gaussian | exact | - | 1.000 | 0.51 | 20,000 | 256 |
| gaussian | IVF-Flat | nprobe=32 | 0.711 | 0.59 | 4,761 | 266 |
| gaussian | HNSW | ef=40 | 0.876 | 0.96 | 1,149 | 354 |
| gaussian | HNSW | ef=160 | 0.971 | 3.20 | 3,278 | 354 |

What to take from it, including the parts that are not flattering:

- **Structure decides everything.** On clustered data IVF-Flat reaches 0.987 recall after scanning 184 of 20,000 vectors. On unstructured Gaussian data the same index needs 4,761 evaluations for 0.711, and HNSW reaches 0.929 at ef=80 with 1,957. Graph methods degrade more gracefully when there are no clusters to prune.
- **HNSW is not faster here.** In distance evaluations HNSW wins by 20-80x over a full scan, but at n=20k and 64 dimensions a single BLAS matrix-vector pass (0.5 ms) is already so cheap that Python-level graph traversal loses on wall clock: exact 0.56 ms vs HNSW 0.32 ms at ef=10 and 2.09 ms at ef=160. The graph's advantage is in evaluation count and it only becomes a latency win at larger n, in compiled code; this implementation is for understanding and correctness, not speed. HNSW also costs 51-72 s to build (IVF-Flat: about 1-2 s) and 360 bytes/vector (vectors plus links).
- **PQ buys memory, not speed or free accuracy.** 29 bytes/vector against 266 (9x), but ADC-only recall plateaus at 0.744 on `clusters` whether you probe 8 cells or 32, because quantisation error, not the missed cells, is the limit. Exact re-ranking the top 4k ADC hits restores 0.994, at the price of keeping the raw vectors (285 bytes/vector), which defeats the memory saving unless the raw vectors live somewhere cheaper (disk). My IVF-PQ is also slower than IVF-Flat here, because it builds a lookup table per probed cell in Python.
- **Filtered search: do not walk the graph for selective filters.** With a filter matching 1% of rows, a graph walk needs 13,430 evaluations (65 ms) while scanning only the matching rows needs 199 (0.07 ms); at 0.4% the walk visits essentially the whole index (143 ms). In evaluation counts the two cross over at about 15% selectivity (walk 2,650 vs exact 4,031 at 20%; 5,220 vs 1,930 at 10%), so `HNSWIndex.brute_force_below` defaults to 0.15. That default was chosen after seeing this one experiment, on one dataset: expect the crossover to move with n, dimension and `ef`. In wall-clock terms exact scans of the matching rows won at *every* selectivity I tried (down from 50%) in this Python implementation. A filter that picks whole clusters is harder for the graph than a random one (recall 0.982 at 25% of rows vs 1.000 for a random 20%).
- **Not tested:** real embeddings, n beyond 20k, concurrent writers, multi-threaded search, or recall under heavy delete/upsert churn (tombstones are never unlinked from the graph; `compact()` rebuilds).

## Design notes

- Rows are positions in an index (0..n-1); the `Collection` maps them to ids. Deleting tombstones a row; the index skips it but HNSW still routes through it.
- Filters become a boolean mask over rows (cached per filter and invalidated by writes) and every index accepts `mask=`, so filtering happens inside the search instead of post-filtering the top-k (which returns fewer than k, or nothing, for selective filters).
- HNSW and IVF are deterministic given a seed. Cosine works by normalising and using squared L2, which has the same ranking.
- Persistence is a single `.npz` of arrays plus JSON metadata, written to a temp file and renamed; loading never unpickles anything.
- Upserts validate the whole batch (shapes, zero vectors under cosine, JSON-serialisable metadata, duplicate ids) before touching any state.

## Layout

```
src/vecdb/   util.py (metrics, k-means, top-k) pq.py flat.py ivf.py hnsw.py filters.py
             collection.py io.py datasets.py bench.py __main__.py
tests/       76 tests: exactness vs NumPy brute force, ADC identity, recall monotonicity,
             HNSW graph invariants (degree bounds, connectivity), filtered search, round trips
experiments/ benchmark.py -> results.md
```

## Roadmap

SQ8 cells inside IVF and `save_index` support (a flat `SQ8Index` is in v0.2); OPQ rotation before PQ; a multi-threaded batch search path; hybrid BM25 + vector scoring on top of `ragkit`; a real-embedding benchmark (needs a download-free source).

## Hybrid search (v0.4)

`HybridCollection` pairs a `Collection` with Okapi BM25 over per-id texts and fuses the two rankings
with reciprocal-rank fusion (`fusion="rrf"`, scale-free) or min-max weighted scores (`fusion="weighted"`,
`alpha` = weight on vector similarity). Metadata filters apply to both retrievers; upserts/deletes keep
the lexical side in sync (BM25 rebuilds lazily). Exact identifiers such as error codes are found by
BM25 even when the embedding is unrelated. Tested on tiny hand-made fixtures only (8 tests); no
retrieval-quality benchmark has been run yet, so no claim is made about fusion gains on real data.

```python
from vecdb import HybridCollection
h = HybridCollection(384)
h.upsert(ids, vectors, metas, texts)
hits = h.query(qvec, "error E1234", k=5, where={"team": "auth"})
```


## Parallel batch search (v0.5)

`parallel_search(index, Q, k, workers=4, shard=64, **kwargs)` shards the query matrix across threads and returns exactly what `index.search` would (row order preserved; works with `mask=`). Gains come only from GIL-releasing BLAS scans; pure-Python HNSW gets none. `last_stats` is unreliable afterwards; don't mutate the index mid-search.

Measured once on a 2-core sandbox (FlatIndex, 50k x 64 float32, 2000 queries, k=10, shard=100): 1.04 s serial, 0.85 s with 2 workers, 0.73 s with 4 (about 1.4x; single run, noisy, not a benchmark suite). 4 new tests (101 total).


## OPQ (v0.6)

`vecdb.opq.OPQ` learns an orthogonal rotation (alternating PQ fit / closed-form Procrustes via SVD) before product quantisation; the query is rotated and ADC is unchanged. `experiments/opq_eval.py` (synthetic, m=8, 6 bits, no re-rank, 8k x 32):

| data | PQ mse | OPQ mse | PQ recall@10 | OPQ recall@10 |
|---|---|---|---|---|
| clustered, axis-aligned | 4.58 | 4.33 | 0.414 | 0.416 |
| clustered, rotated + anisotropic | 10.89 | 8.99 | 0.460 | 0.503 |

Gain is real but modest and concentrated where variance is not axis-aligned; on the axis-aligned set recall is a tie. Not yet wired into `IVFIndex`.


## IVF-SQ8 (v0.8)

`IVFIndex(dim, nlist=64, sq8=True)` stores each cell as one uint8 per dimension (global per-dimension affine codes fitted at train time), 4x smaller than IVF-Flat; `keep_raw=True` enables `refine=`. Mutually exclusive with `pq_m`. Vectors outside the training range are clipped. Persisted by `save_index`/`load_index`. Tests: `tests/test_ivf_sq8.py`.


## Hybrid fusion benchmark (v0.9)

`PYTHONPATH=src python experiments/hybrid_eval.py` — 1,500 synthetic docs, 300 queries/seed, 3 seeds, one relevant doc per query.
"semantic" queries share only same-topic words with the target (vector near its topic); "lookup" queries hold the target's rare ID with a noisy vector.

| mode | MRR semantic | MRR lookup | MRR all | recall@10 semantic | recall@10 lookup |
|---|---|---|---|---|---|
| dense | 0.058 | 0.018 | 0.038 | 0.184 | 0.073 |
| bm25 | 0.000 | 1.000 | 0.500 | 0.000 | 1.000 |
| rrf | 0.000 | 0.828 | 0.414 | 0.000 | 1.000 |
| weighted a=0.5 | 0.026 | 0.828 | 0.427 | 0.138 | 1.000 |
| weighted a=0.2 | 0.018 | 1.000 | 0.509 | 0.116 | 1.000 |

Honest reading: the semantic task is hard by construction (~50 same-topic docs, the target is indistinguishable by topic alone), so absolute numbers are low. Fusion did **not** beat the best single retriever on the mixed set: BM25 alone is best on lookups, and fusion only dilutes the dense signal on semantic queries because BM25 ranks same-topic distractors highly (RRF recall 0 there). alpha=0.2 was picked after seeing the table. Synthetic data only; a real-embedding benchmark is still open.

### Diversity re-ranking (MMR)

`HybridCollection.query_mmr(vector, text, k, lambda_=0.7)` re-ranks the top `3*k` fused hits with Maximal Marginal Relevance (relevance = scaled fused score, redundancy = max cosine to already-picked hits). `lambda_=1` equals `query`; lower values suppress near-duplicates. Unit-tested on a duplicate-pair case; no retrieval-quality benchmark yet.

### BM25 parameters

`HybridCollection(dim, bm25_k1=1.5, bm25_b=0.75)` makes the BM25 saturation and length-normalisation parameters configurable (validated, persisted in `bm25.json`; older saves load with defaults). `b=0` removes length normalisation. Not tuned on any benchmark here.

### Explaining hybrid results

`HybridCollection.explain(vector, text, k)` runs the RRF query and returns, per hit, the fused score, its rank in the dense and BM25 lists (None if absent), the raw distance/BM25 score, and whether it came from `both`, `dense` or `lexical`. Useful for debugging why a document surfaced.
