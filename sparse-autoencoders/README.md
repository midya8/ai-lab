# sparse-autoencoders

Sparse autoencoders in NumPy, trained on synthetic activations built from a known dictionary of superposed features. With real model activations you can only ask whether the latents look sensible. Here I can check whether the SAE actually recovered the features, and what that cost in sparsity and reconstruction.

Five variants with hand-written backprop (checked against finite differences): ReLU + L1, TopK, BatchTopK, JumpReLU and Gated.

```bash
cd sparse-autoencoders
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                         # 36 tests, ~3 s
cd experiments && OMP_NUM_THREADS=1 python frontier.py     # ~9 min
```

```python
import numpy as np
from sparse_ae import SAE, fit, make_world, feature_recovery, fvu, l0

world = make_world(n_features=64, d=32, l0=3, seed=0)   # 64 features squeezed into 32 dims
sae = SAE(d=32, n_latents=128, mode="batchtopk", k=3)
fit(sae, world, steps=3000)
x, a = world.sample(4096, np.random.default_rng(1))
z = sae.encode(x)
print(feature_recovery(world.directions, sae.params["W_dec"]), fvu(x, sae.decode(z)), l0(z))
```

## Main comparison

64 true features in 32 dims, true L0 = 3, 128 latents, 5 seeds:

| method | L0 | FVU | features matched | purity of live latents |
|---|---|---|---|---|
| ReLU, l1=0.2 | 9.5 | 0.077 | 1.00 | 0.50 |
| TopK, k=3 | 2.9 | 0.155 | 1.00 | 0.78 |
| BatchTopK, k=3 | 3.0 | 0.083 | 1.00 | 0.93 |
| JumpReLU, 0.05 | 3.1 | 0.084 | 1.00 | 0.77 |
| Gated, l1=0.8 | 4.8 | 0.083 | 0.98 | 0.49 |

Every method finds the features once it's sparse enough. BatchTopK halves TopK's reconstruction error at the same L0, because real samples have anywhere from 0 to 8 active features and a fixed k cuts off the busy ones. L1 needs about 2.4x as many active latents for the same FVU and shrinks activations to ~65% of their true size. JumpReLU matches BatchTopK on the frontier but keeps more junk latents alive.

## Other findings

- With up to 8x more features than dimensions and true L0 <= 6, TopK and BatchTopK recovered 100% of features. In the dense regime (L0 12-24), reconstruction stays good while recovery collapses (as low as 4%). So good reconstruction doesn't mean you found the features.
- Dead-latent resampling did nothing here. The dead latents were just surplus capacity.
- Rare features (under 1.3% firing rate) were recovered as well as common ones.
- Feature absorption in a parent/child hierarchy grows with sparsity pressure for every architecture. ReLU + L1 absorbs about twice as much as Gated at similar L0. My absorption metric also breaks when child and parent directions are correlated, which is worth knowing.

All of this is independent synthetic features with 3-5 seeds, so the rankings are suggestive, not numbers that transfer to real models. Raw output is in `experiments/*_results.txt`.
