# gnnlab

Graph neural networks from scratch in NumPy — **GCN, GraphSAGE and multi-head GAT with hand-written backprop, checked against finite differences** — plus two small, honest studies of *when graphs help* and *what depth does to them*. Dense float64 on a few hundred nodes: readable rather than fast.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q          # 55 tests, ~3 s
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/homophily.py   # ~5 min on 2 cores
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/depth.py       # ~6 min
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/decoupled_depth.py      # ~3 min
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/decoupled_homophily.py # ~1 min
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
| `APPNP` | MLP, then K steps of personalised-PageRank propagation `H <- (1-a) Ahat H + a Z` (parameter-free; adjoint is tested) |
| `SGC` | `(Ahat^K X) W + b`: K-hop smoothing cached once, then a linear classifier |
| `JKNet` | GCN stack whose classifier reads the concatenation of *all* layer outputs (jumping knowledge) |
| `label_propagation` | Zhou et al. label spreading from the training labels; edges only, no features |
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

## Study 3 (v0.2): does decoupling propagation from transformation fix depth?

Study 2 left a caveat: plain GCN's collapse mixes oversmoothing with deep-network trainability. SGC separates them cleanly: it is a *linear* model on `Ahat^K X`, nothing deep to train, so if it fails at large K that is the smoothing itself. Same graphs as Study 2 (h = 0.8, 4 seeds, width 32, 200 epochs; test acc, train acc in brackets; `experiments/decoupled_depth_results.txt`):

| model | K/L=2 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| GCN, plain (L layers) | 0.99 (1.00) | 0.98 (1.00) | **0.23** (0.28) | **0.24** (0.32) | not run |
| JKNet, concat (L layers) | 0.98 (1.00) | 0.99 (1.00) | 0.99 (1.00) | 0.99 (1.00) | not run |
| SGC (K hops) | 0.99 (1.00) | 0.99 (1.00) | 0.86 (0.91) | **0.26** (0.33) | **0.23** (0.29) |
| APPNP, a = 0.1 (K hops) | 0.98 (0.99) | 0.98 (1.00) | 0.97 (1.00) | 0.97 (1.00) | 0.97 (1.00) |
| APPNP, a = 0 (K hops) | 0.98 (1.00) | 0.99 (1.00) | 0.99 (1.00) | **0.36** (0.38) | **0.24** (0.25) |

* **Decoupling alone does not help; teleportation does.** SGC and APPNP with a = 0 have *no* trainable depth and still fall to chance (0.25) by K = 32-64, with train accuracy collapsing too: the features really are smoothed into one direction. So at least part of plain GCN's collapse in Study 2 is genuine oversmoothing, not just trainability.
* APPNP with a = 0.1 loses nothing from K = 2 to K = 64: mixing a fixed share of each node's own prediction back in at every hop stops the fixed point from being the degenerate one (the output converges to the closed form `a (I - (1-a) Ahat)^-1 Z`, which is tested).
* JKNet is the other fix: shallow, unsmoothed layers stay visible to the classifier, so extra layers cannot hurt. (Not run beyond L = 32 for GCN/JKNet because of the cost of deep dense stacks.)
* At L = 2..8 everything sits at the ceiling (0.98-0.99): depth buys no accuracy on this graph, only risk.

### Homophily again, with decoupled models and label propagation

Same graphs, splits and seeds as Study 1, so the MLP and GCN columns are identical to that table (`experiments/decoupled_homophily_results.txt`):

| edge homophily | MLP | GCN | SGC k=2 | APPNP k=10 | label propagation |
|---|---|---|---|---|---|
| 0.1 | 0.590 | 0.407 | 0.404 | 0.505 | 0.107 |
| 0.2 | 0.590 | 0.379 | 0.384 | 0.547 | 0.176 |
| 0.3 | 0.590 | 0.478 | 0.468 | 0.595 | 0.287 |
| 0.5 | 0.590 | 0.751 | 0.731 | 0.732 | 0.577 |
| 0.7 | 0.590 | 0.945 | 0.936 | 0.902 | 0.897 |
| 0.9 | 0.590 | 0.998 | 0.998 | 0.998 | 0.998 |

(std 0.002-0.067 over 6 graphs; see the results file.)

* SGC tracks GCN almost cell for cell: on this data the non-linearity buys nothing.
* APPNP's retained own-prediction makes it the most robust graph model at low homophily (0.55 vs GCN 0.38 at h = 0.2) but it still trails the MLP below h = 0.3, and it is ~0.03-0.04 behind GCN at 0.7 (K = 10, a = 0.1 was not tuned for either regime).
* Label propagation, which never touches a feature, matches the feature-aware models at h = 0.9 and is competitive at 0.7, then falls **below the 0.25 chance level** at h <= 0.2: with mostly cross-class edges, spreading labels actively spreads the *wrong* ones. At h = 0.1 it scores 0.107.

## Limitations

* Synthetic SBM graphs with Gaussian features; no real citation graphs (no download-free source). Effects such as GAT's weak showing may change on real data with more labels.
* Dense `n x n` operators: fine up to a few thousand nodes, not beyond. GAT is ~30x slower than GCN here.
* No dropout, no mini-batching/neighbour sampling, no edge features.
* Seeds are few (6 and 4); differences under ~0.03 between columns are within noise.

## Ideas next

dropout; hyperparameter sweeps for APPNP (K, a) per homophily regime; a GCN on real text graphs built from `docqa`/`ragkit` chunk similarity; sparse (CSR) operators; heterophily-aware layers (H2GCN-style ego/neighbour/2-hop separation).
