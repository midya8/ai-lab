# gnnlab

Graph neural networks from scratch in NumPy — **GCN, GraphSAGE and multi-head GAT with hand-written backprop, checked against finite differences** — plus two small, honest studies of *when graphs help* and *what depth does to them*. Dense float64 on a few hundred nodes: readable rather than fast.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q          # 36 tests, ~3 s
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/homophily.py   # ~5 min on 2 cores
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/depth.py       # ~6 min
```

```python
from gnnlab import GNN, sbm_graph, build_ops, make_split, fit

g = sbm_graph(n=400, k=4, homophily=0.8, seed=0)      # attributed stochastic block model
ops = build_ops(g.adj)                                 # Ahat, neighbour-mean, attention mask
train, val, test = make_split(g.y, per_class=20, n_val=100)
model = GNN("gcn", in_dim=16, hidden=32, out_dim=4, depth=2)   # "mlp" | "gcn" | "sage" | "gat"
print(fit(model, g.x, g.y, ops, train, val, test).test_acc)
```

## What is in the box

| Piece | Notes |
|---|---|
| `GCNLayer` | `Ahat H W + b`, `Ahat = D~^-1/2 (A+I) D~^-1/2` |
| `SAGELayer` | mean aggregator with *separate* self and neighbour weights |
| `GATLayer` | multi-head additive attention over `N(i)+{i}`, full softmax backward (leaky-ReLU, masked) |
| `GNN` | stack of any of the above (or an MLP that ignores the graph), optional residual connections and **PairNorm** (with its own backward) |
| `sbm_graph` | attributed SBM with a *target* edge homophily and fixed expected degree, so features carry the same signal whatever the edges do |
| `fit` | full-batch Adam, L2 on weights, best-validation-epoch restore (test accuracy is read only there) |
| `dirichlet_energy`, `mean_pairwise_cosine` | oversmoothing diagnostics |

Correctness is tested, not asserted: every layer type, depth 1 and 3, with residual + PairNorm, multi-head GAT and isolated nodes, all match central finite differences; GAT attention rows are distributions supported on neighbours; GCN/SAGE/GAT are **permutation-equivariant** (relabel the nodes, outputs permute); the MLP provably ignores the graph; repeated `Ahat` propagation drives Dirichlet energy monotonically to ~0 on a connected graph.

## Study 1: graphs help only if edges agree with labels

4 classes, 400 nodes, 16-d Gaussian features (the same weak signal everywhere), mean degree 10, 20 labels per class, 2 layers, hidden 32, mean ± std over 6 graphs (`experiments/homophily_results.txt`):

| edge homophily | MLP | GCN | GraphSAGE | GAT (4 heads) |
|---|---|---|---|---|
| 0.1 | 0.590 ± 0.067 | 0.407 ± 0.023 | 0.614 ± 0.047 | 0.345 ± 0.028 |
| 0.2 | 0.590 | 0.379 ± 0.041 | 0.565 ± 0.051 | 0.303 ± 0.028 |
| 0.3 | 0.590 | 0.478 ± 0.050 | 0.568 ± 0.038 | 0.383 ± 0.072 |
| 0.5 | 0.590 | 0.751 ± 0.053 | 0.706 ± 0.050 | 0.601 ± 0.044 |
| 0.7 | 0.590 | 0.945 ± 0.024 | 0.869 ± 0.037 | 0.893 ± 0.033 |
| 0.9 | 0.590 | 0.998 ± 0.002 | 0.973 ± 0.020 | 0.987 ± 0.008 |

(The MLP row is one number because it never sees the edges and the graphs share seeds; "homophily 0.25" is the no-information point for four classes.)

* At homophily 0.7 and above the edges are worth +35 to +41 accuracy points over the MLP, and GCN's fixed averaging is the best model; at 0.5 it is already +16.
* At homophily 0.2, GCN falls **21 points under the MLP**: it averages a node with neighbours that mostly carry the wrong class. GraphSAGE, which keeps the node's own features in a separate weight matrix, can learn to ignore the neighbourhood and stays at MLP level. That is the textbook argument for self/neighbour separation, and here it is measured.
* GAT is *worst* at low homophily in this run. Plausible reasons (not tested here): only 80 labels to learn attention from, and more parameters. This is a small-data result on synthetic graphs; it is not a claim about GAT in general.

## Study 2: depth, oversmoothing and two fixes

GCN on an h = 0.8 graph, hidden 32, 200 epochs, 4 seeds; each cell is test acc (train acc), Dirichlet energy and mean pairwise cosine of the last hidden layer (`experiments/depth_results.txt`):

| variant | L=2 | L=8 | L=16 | L=32 |
|---|---|---|---|---|
| plain | 0.99 (1.00) · 0.160 · 0.52 | 0.98 (1.00) · 0.064 · 0.69 | **0.23 (0.28)** · 0.009 · 0.99 | **0.24 (0.32)** · 0.007 · 1.00 |
| residual | 0.99 · 0.160 · 0.52 | 0.98 · 0.037 · 0.84 | 0.99 (1.00) · 0.016 · 0.94 | 0.98 (0.99) · 0.040 · 0.90 |
| PairNorm | 0.99 · 0.224 · 0.30 | 0.99 · 0.166 · 0.26 | 0.99 · 0.160 · 0.26 | 0.99 · 0.162 · 0.26 |
| residual + PairNorm | 0.99 · 0.224 · 0.30 | 0.99 · 0.151 · 0.34 | 0.99 · 0.128 · 0.47 | 1.00 · 0.115 · 0.57 |

* Plain GCN collapses between 8 and 16 layers: the hidden layer becomes one direction (cosine ≈ 1, energy ≈ 0.01), and **train** accuracy collapses too, so this is a failure to fit, not overfitting.
* Honest caveat: the plain-model collapse mixes oversmoothing with plain deep-network trainability (vanishing signal through 16 ReLU layers with 1/√fan-in Glorot init). The residual variant recovers even though its cosine stays high (0.94), so cosine/energy are *symptoms that correlate with failure here*, not proofs of cause.
* PairNorm keeps energy near 0.16 and cosine near 0.26 at every depth; residual + PairNorm scores highest at L=32 (1.00 vs 0.98-0.99), a gap within noise at 4 seeds.
* At L=2 the task is already at ceiling (0.99), so depth has nothing to add on this graph — depth is not helping accuracy anywhere in this table.

## Limitations

* Synthetic SBM graphs with Gaussian features; no real citation graphs (no download-free source). Effects such as GAT's weak showing may change on real data with more labels.
* Dense `n x n` operators: fine up to a few thousand nodes, not beyond. GAT is ~30x slower than GCN here.
* No dropout, no mini-batching/neighbour sampling, no edge features.
* Seeds are few (6 and 4); differences under ~0.03 between columns are within noise.

## Ideas next

APPNP / SGC / JKNet to test whether decoupling propagation from transformation removes the depth collapse; label-propagation baseline; a GCN on real text graphs built from `docqa`/`ragkit` chunk similarity; sparse (CSR) operators; heterophily-aware layers (H2GCN-style ego/neighbour/2-hop separation).
