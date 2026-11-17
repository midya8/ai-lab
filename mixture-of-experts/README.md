# mixture-of-experts

Sparse MoE routing in NumPy with hand-written backprop (checked against finite differences): top-k token-choice and expert-choice routing, capacity limits with token dropping, the Switch balance loss, router z-loss, DeepSeek-style aux-loss-free bias balancing, and shared experts.

The task is a set of Gaussian clusters, each with its own target function, so specialising is the right move and I can measure whether the router actually learns it.

```bash
cd mixture-of-experts
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                       # 38 tests, ~1 s
cd experiments && OMP_NUM_THREADS=1 python collapse.py   # ~4 min
```

```python
from moe import MoELayer, Net, fit, make_task, sample_task, load_stats

task = make_task(n_clusters=8, seed=0)
x, y, _ = sample_task(task, 1024, seed=1)
xt, yt, _ = sample_task(task, 1024, seed=2)
layer = MoELayer(d=32, f=32, n_experts=8, k=2, capacity_factor=1.25, aux_coef=0.01, seed=0)
res = fit(Net(d_in=8, d_out=4, layer=layer, d=32, seed=0), x, y, xt, yt, steps=400, lr=3e-3)
print(res.test_mse, load_stats(res.load), res.drop_rate)
```

## What I learned

- **Top-1 routing collapses without a balance loss.** About 4 of 8 experts die and 40% of tokens get dropped. An aux coefficient of 0.1 fixes it and cuts error 4.7x. Top-2 barely needs the loss because the second choice rescues tokens.
- **Balanced isn't the same as specialised.** When the clusters are skewed, pushing for balance splits the popular cluster across experts, and routing quality drops even as MSE improves.
- **Capacity:** top-1 pays for every dropped token. Top-2 still does fine with 29% of slots dropped.
- **Sparse vs dense:** top-2 MoE beats a dense FFN with the same active compute by 1.6-2.1x. Against a dense FFN with the same total parameters it mostly ties. So sparsity buys you equal quality for less compute, not better quality at equal size.
- **Bias balancing (no aux loss)** gets top-2 to load CV 0.06 with zero dropped tokens, and keeps more specialisation on skewed data (NMI 0.82 vs 0.70 with the aux loss).
- **A shared expert didn't help here.** That's expected, since this task has nothing common across clusters for it to absorb. Finer-grained experts helped slightly.

Seed counts are 6-8 and many differences are within one std. Full tables are in `experiments/*_results.txt`.

This is about routing mechanics on toy regression, not language-model scale. Routing loops over tokens in Python, so it's clear rather than fast.
