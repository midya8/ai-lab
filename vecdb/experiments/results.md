# vecdb benchmark results

Synthetic data, k=10, 200 queries, ground truth = exact search. Single-process Python; latency is per single-query call and includes interpreter overhead. Bytes/vector counts the index structures (ids, centroids, codebooks, graph links at 4 bytes/link), not a Python process.


## clusters (n=20000, dim=64)

| method | setting | recall@k | ms/query | distance evals/query | bytes/vector | build (s) |
|---|---|---|---|---|---|---|
| flat (exact) | - | 1.000 | 0.56 | 20000 | 256 | 0.0 |
| IVF-Flat nlist=141 | nprobe=1 | 0.877 | 0.07 | 89 | 266 | 1.2 |
| IVF-Flat nlist=141 | nprobe=2 | 0.987 | 0.09 | 184 | 266 | 1.2 |
| IVF-Flat nlist=141 | nprobe=4 | 0.998 | 0.13 | 600 | 266 | 1.2 |
| IVF-Flat nlist=141 | nprobe=8 | 0.999 | 0.20 | 1179 | 266 | 1.2 |
| IVF-Flat nlist=141 | nprobe=16 | 1.000 | 0.33 | 2900 | 266 | 1.2 |
| IVF-Flat nlist=141 | nprobe=32 | 1.000 | 0.60 | 5342 | 266 | 1.2 |
| IVF-PQ m=16 | nprobe=8, ADC only | 0.744 | 0.48 | 1179 | 29 | 8.1 |
| IVF-PQ m=16 | nprobe=8, ADC + exact re-rank x4 | 0.994 | 0.52 | 1179 | 285 | 8.1 |
| IVF-PQ m=16 | nprobe=8, ADC + exact re-rank x16 | 0.999 | 0.54 | 1179 | 285 | 8.1 |
| IVF-PQ m=16 | nprobe=32, ADC only | 0.744 | 1.72 | 5342 | 29 | 8.1 |
| IVF-PQ m=16 | nprobe=32, ADC + exact re-rank x4 | 0.994 | 1.76 | 5342 | 285 | 8.1 |
| IVF-PQ m=16 | nprobe=32, ADC + exact re-rank x16 | 1.000 | 1.79 | 5342 | 285 | 8.1 |
| HNSW M=16 efC=100 | ef=10 | 0.992 | 0.32 | 236 | 360 | 51.3 |
| HNSW M=16 efC=100 | ef=20 | 1.000 | 0.47 | 308 | 360 | 51.3 |
| HNSW M=16 efC=100 | ef=40 | 1.000 | 0.71 | 375 | 360 | 51.3 |
| HNSW M=16 efC=100 | ef=80 | 1.000 | 1.17 | 513 | 360 | 51.3 |
| HNSW M=16 efC=100 | ef=160 | 1.000 | 2.09 | 985 | 360 | 51.3 |


## gaussian (n=20000, dim=64)

| method | setting | recall@k | ms/query | distance evals/query | bytes/vector | build (s) |
|---|---|---|---|---|---|---|
| flat (exact) | - | 1.000 | 0.51 | 20000 | 256 | 0.0 |
| IVF-Flat nlist=141 | nprobe=1 | 0.171 | 0.07 | 149 | 266 | 1.8 |
| IVF-Flat nlist=141 | nprobe=2 | 0.207 | 0.09 | 285 | 266 | 1.8 |
| IVF-Flat nlist=141 | nprobe=4 | 0.278 | 0.12 | 576 | 266 | 1.8 |
| IVF-Flat nlist=141 | nprobe=8 | 0.387 | 0.19 | 1134 | 266 | 1.8 |
| IVF-Flat nlist=141 | nprobe=16 | 0.524 | 0.32 | 2369 | 266 | 1.8 |
| IVF-Flat nlist=141 | nprobe=32 | 0.711 | 0.59 | 4761 | 266 | 1.8 |
| IVF-PQ m=16 | nprobe=8, ADC only | 0.347 | 0.47 | 1134 | 29 | 8.4 |
| IVF-PQ m=16 | nprobe=8, ADC + exact re-rank x4 | 0.387 | 0.50 | 1134 | 285 | 8.4 |
| IVF-PQ m=16 | nprobe=8, ADC + exact re-rank x16 | 0.387 | 0.53 | 1134 | 285 | 8.4 |
| IVF-PQ m=16 | nprobe=32, ADC only | 0.493 | 1.69 | 4761 | 29 | 8.4 |
| IVF-PQ m=16 | nprobe=32, ADC + exact re-rank x4 | 0.685 | 1.73 | 4761 | 285 | 8.4 |
| IVF-PQ m=16 | nprobe=32, ADC + exact re-rank x16 | 0.711 | 1.76 | 4761 | 285 | 8.4 |
| HNSW M=16 efC=100 | ef=10 | 0.696 | 0.39 | 449 | 354 | 71.5 |
| HNSW M=16 efC=100 | ef=20 | 0.795 | 0.59 | 676 | 354 | 71.5 |
| HNSW M=16 efC=100 | ef=40 | 0.876 | 0.96 | 1149 | 354 | 71.5 |
| HNSW M=16 efC=100 | ef=80 | 0.929 | 1.73 | 1957 | 354 | 71.5 |
| HNSW M=16 efC=100 | ef=160 | 0.971 | 3.20 | 3278 | 354 | 71.5 |


## Filtered search on HNSW (clusters, ef=64; unfiltered exact scan = 0.57 ms/query)

`walk` traverses the graph and collects only matching nodes; `exact` scans just the matching rows.

| filter | matching rows | walk recall | walk ms | walk evals | exact ms | exact evals | faster |
|---|---|---|---|---|---|---|---|
| random 50.0% | 49.58% | 1.000 | 1.69 | 679 | 1.49 | 9916 | exact |
| random 30.0% | 29.73% | 1.000 | 2.75 | 1620 | 0.92 | 5946 | exact |
| random 20.0% | 20.16% | 1.000 | 4.34 | 2650 | 0.68 | 4031 | exact |
| random 10.0% | 9.65% | 1.000 | 9.57 | 5220 | 0.29 | 1930 | exact |
| random 5.0% | 5.10% | 1.000 | 16.17 | 8947 | 0.15 | 1019 | exact |
| random 2.0% | 1.82% | 1.000 | 43.37 | 15458 | 0.08 | 364 | exact |
| random 0.5% | 0.41% | 1.000 | 143.03 | 20075 | 0.06 | 83 | exact |
| 1 cluster | 1.00% | 1.000 | 65.38 | 13430 | 0.07 | 199 | exact |
| 5 clusters | 4.88% | 0.997 | 25.01 | 8789 | 0.15 | 975 | exact |
| 25 clusters | 24.99% | 0.982 | 4.70 | 459 | 0.86 | 4998 | exact |
