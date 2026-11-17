# gnn

GCN, GraphSAGE, GAT, APPNP, SGC and JKNet in NumPy with hand-written backprop (checked against finite differences), plus label propagation as a no-features baseline. Graphs come from an attributed stochastic block model where I can dial the edge homophily, so I could ask two questions: when do edges actually help, and what goes wrong as you stack more layers?

```bash
cd gnn
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                                   # 55 tests, ~3 s
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/homophily.py     # ~5 min
OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/depth.py         # ~6 min
```

```python
from gnn import GNN, sbm_graph, build_ops, make_split, fit

g = sbm_graph(n=400, k=4, homophily=0.8, seed=0)
ops = build_ops(g.adj)
train, val, test = make_split(g.y, per_class=20, n_val=100)
model = GNN("gcn", in_dim=16, hidden=32, out_dim=4, depth=2)    # "mlp" | "gcn" | "sage" | "gat"
print(fit(model, g.x, g.y, ops, train, val, test).test_acc)
```

## When do graphs help?

Test accuracy, 4 classes, mean over 6 graphs:

| edge homophily | MLP | GCN | GraphSAGE | GAT |
|---|---|---|---|---|
| 0.2 | 0.590 | 0.379 | 0.565 | 0.303 |
| 0.5 | 0.590 | 0.751 | 0.706 | 0.601 |
| 0.7 | 0.590 | 0.945 | 0.869 | 0.893 |
| 0.9 | 0.590 | 0.998 | 0.973 | 0.987 |

With mostly same-class edges, GCN is +35 to +41 points over the MLP. With mostly cross-class edges it drops 21 points *below* the MLP, while GraphSAGE (which keeps separate weights for the node itself) just learns to ignore the neighbours. GAT did worst at low homophily here, probably because 80 labels isn't much to learn attention from.

## Depth

Plain GCN on an easy graph is fine at 8 layers and collapses to chance at 16, train accuracy included. Some of that is just deep-net trainability, so I checked with SGC, which is linear with nothing deep to train: it also collapses by K=32 hops. So the oversmoothing is real.

What fixes it: APPNP with teleport 0.1 holds 0.97 all the way to K=64, and JKNet (classifier sees every layer) stays at 0.99. Residual connections and PairNorm also work. But on this graph 2 layers already hit 0.99, so depth never bought any accuracy, only risk.

Full tables are in `experiments/*_results.txt`.

## Limitations

Synthetic graphs only, dense n x n matrices (fine to a few thousand nodes), no dropout or minibatching, and only 4-6 seeds, so gaps under ~0.03 are noise.
