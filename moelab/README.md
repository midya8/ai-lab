# moelab

Sparse **Mixture-of-Experts routing from scratch in NumPy**: token-choice top-k and expert-choice routing, capacity limits with token dropping, the Switch load-balancing loss and the ST-MoE router z-loss — all with hand-written backprop **checked against finite differences** — plus three seeded studies of the things that actually go wrong in MoE training: routing collapse, capacity drops, and whether sparsity beats a dense FFN.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                              # 25 tests, ~1 s
cd experiments && OMP_NUM_THREADS=1 python collapse.py          # ~4 min on 2 cores
OMP_NUM_THREADS=1 python capacity.py; OMP_NUM_THREADS=1 python compare.py   # ~3 min each
```

```python
import numpy as np
from moelab import MoELayer, Net, fit, make_task, sample_task, load_stats

task = make_task(n_clusters=8, seed=0)                  # 8 clusters, each with its own target function
x, y, z = sample_task(task, 1024, seed=1)
xt, yt, zt = sample_task(task, 1024, seed=2)
layer = MoELayer(d=32, f=32, n_experts=8, k=2, capacity_factor=1.25, aux_coef=0.01, seed=0)
net = Net(d_in=8, d_out=4, layer=layer, d=32, seed=0)
res = fit(net, x, y, xt, yt, steps=400, lr=3e-3)
print(res.test_mse, load_stats(res.load), res.drop_rate)
```

## What is in the box

| Piece | Notes |
|---|---|
| `MoELayer(routing="topk")` | softmax router, top-k experts per token, gates renormalised over the chosen k (raw probability for k=1 so the router still gets a task gradient), per-expert capacity `ceil(cf·N·k/E)` with GShard-style priority (all first choices before any second choice), overflow tokens skip the layer via the residual |
| `MoELayer(routing="expert_choice")` | every expert picks its top-`capacity` tokens: perfect balance by construction, but some tokens get no expert and some get several |
| `aux_coef`, `z_coef` | Switch load-balance loss `E·Σ f_i P_i` (hard top-1 fractions × mean probabilities) and router z-loss `mean(logsumexp²)`, both with their own backward |
| `jitter` | multiplicative router-input noise, train mode only |
| `DenseFFN` | same interface and residual, for parameter- or compute-matched baselines |
| `make_task` / `sample_task` | K Gaussian clusters, each with its own target `tanh(1.5·x W_k)` — a task where specialising is the right thing to do, so routing quality is measurable (`routing_nmi`, `routing_purity` vs the latent cluster) |
| `load_stats` | load CV, max/mean, dead experts (<10% of fair share), normalised entropy |

The discrete decisions (which experts, which tokens survive capacity) are constants in the backward pass; gradients flow through the continuous gate weights, as in real implementations. Tests: every routing mode (k=1/2, with and without real token dropping, with and without gate renormalisation, aux + z losses, expert-choice) matches central finite differences — and deliberately breaking the gate backward makes them fail; GShard priority ordering; dropped tokens pass through the residual unchanged; the balance loss equals 1 when uniform and `E` at total collapse; its gradient measurably rebalances a skewed router; expert-choice load is exactly uniform; NMI/purity behave on known partitions.

All studies: 8 experts (`f=32` each, model width 32), N=1024 train / 1024 fresh test points, 400 full-batch Adam steps, mean ± std over seeds that change the task, data and initialisation. Metrics are measured on the test set in eval mode. Raw output is in `experiments/*_results.txt`.

## Study 1: collapse and the balance loss (`collapse.py`, 8 seeds, capacity factor 1.25)

Uniform clusters, test MSE (lower is better):

| k | aux 0 | aux 0.01 | aux 0.1 | aux 1.0 |
|---|---|---|---|---|
| 1 | 0.0356±0.0107 | 0.0175±0.0102 | **0.0075±0.0017** | 0.0103±0.0026 |
| 2 | 0.0063±0.0021 | **0.0059±0.0020** | 0.0077±0.0031 | 0.0074±0.0031 |

* **Top-1 without a balance loss collapses**: load CV 1.13, ~3.8 of 8 experts dead, 40% of tokens dropped for exceeding capacity. `aux=0.1` brings CV to 0.11, no dead experts, ~0% drops, and a 4.7× lower error.
* **Too much is worse**: `aux=1.0` is slightly worse than 0.1 for top-1 (0.0103 vs 0.0075) and routing NMI falls from 0.84 to 0.77.
* **Top-2 barely needs it**: even at aux 0 only ~0.8 experts die (CV 0.44) because the second choice rescues tokens a full expert refuses; the balance loss is flat-to-slightly-harmful above 0.01 (differences are within about one std).
* **Skewed clusters (Zipf(1) frequencies) expose the trade-off**: balancing a router whose data is genuinely imbalanced splits the popular cluster across experts, so routing NMI against the latent cluster *drops* with the coefficient (top-1: 0.72 → 0.64 at aux 0.1, top-2: 0.84 → 0.70) even while MSE improves. Balanced ≠ specialised.

## Study 2: capacity factor (`capacity.py`, 6 seeds, aux 0.01)

| capacity factor | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | unlimited |
|---|---|---|---|---|---|---|---|
| top-1 drop rate | 0.50 | 0.26 | 0.16 | 0.09 | 0.02 | 0.03 | 0 |
| top-1 test MSE | 0.0424 | 0.0272 | 0.0209 | 0.0194 | 0.0102 | 0.0129 | 0.0093 |
| top-2 drop rate | 0.51 | 0.29 | 0.11 | 0.02 | 0.003 | 0 | 0 |
| top-2 test MSE | 0.0209 | 0.0088 | 0.0084 | 0.0059 | 0.0074 | 0.0058 | 0.0057 |

Top-1 pays for every dropped token; top-2 is nearly free of capacity pressure down to cf≈0.75 (29% of slots dropped, MSE 0.0088 vs 0.0057) because most tokens keep their other expert. Non-monotone cells (top-1 cf 1.5 vs 2.0, top-2 cf 1.25 vs 1.5) are inside seed noise.

## Study 3: sparse vs dense as the task gets more clusters (`compare.py`, 6 seeds)

| clusters | top-1 MoE | top-2 MoE | expert-choice (k=2) | dense, 2× wide | dense, 8× wide |
|---|---|---|---|---|---|
| 4 | 0.0046±0.0031 | 0.0046±0.0028 | 0.0116±0.0067 | 0.0051±0.0020 | **0.0025±0.0004** |
| 8 | 0.0108±0.0102 | **0.0059±0.0023** | 0.0088±0.0022 | 0.0124±0.0014 | 0.0063±0.0018 |
| 16 | 0.0278±0.0072 | 0.0135±0.0023 | 0.0175±0.0042 | 0.0222±0.0011 | **0.0131±0.0017** |
| 32 | 0.0472±0.0071 | **0.0242±0.0015** | 0.0271±0.0034 | 0.0394±0.0037 | 0.0323±0.0162 |

Layer parameters touched per token: top-2 MoE ≈ 4.5k (router + 2 experts), dense 2× ≈ 4.2k, dense 8× ≈ 16.9k (same *total* as the MoE, 17.1k–17.6k whole-net). So the fair reading is:

* **Top-2 MoE vs the compute-matched dense FFN**: 1.6–2.1× lower error at 8, 16 and 32 clusters, at about the same active parameters.
* **Top-2 MoE vs the parameter-matched dense FFN**: a tie at 8 and 16 clusters, and ahead at 32 (0.0242 vs 0.0323, but the dense std is 0.016 — one or two bad dense seeds). With 4 clusters the small dense model is already enough and the *wide* dense net wins. Sparsity buys equal quality at ~4× fewer active FFN parameters; it does not beat dense at equal total size.
* **Top-1 is fragile** (large seed variance) and expert-choice, though perfectly balanced, loses to token-choice top-2 here (not investigated why; a plausible cause is that it leaves some tokens with no expert at all).

## Caveats

* Synthetic regression with cluster-separated inputs, tiny models, full batches: this studies routing *mechanics*, not language-model scale behaviour. Expert-choice routing looks across tokens in a batch, which would leak future information in a causal LM.
* Routing runs in Python loops over tokens for capacity bookkeeping — clear, not fast.
* Aux/capacity settings were chosen from the sweeps above on the same task family, so the "best" cells are mildly tuned; the stds are across seeds, not a held-out protocol.
