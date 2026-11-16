# saelab

**Sparse autoencoders (SAEs) from scratch in NumPy, scored on data whose true features are known.** Interpretability work on real language models can only ask "do the latents look sensible?". Here the activations are built from a known dictionary of superposed features, so the question becomes measurable: *did the SAE recover the features, and at what cost in sparsity and fidelity?*

Five SAE flavours with hand-written backprop (**checked against finite differences**): ReLU + L1 penalty, per-sample **TopK**, **BatchTopK** (k latents per sample *on average*, with a learned inference threshold), and (v0.2) **JumpReLU** (learned per-latent thresholds, L0 penalty, straight-through threshold gradient) and **Gated** SAEs (separate gate and magnitude paths). Six seeded studies, raw output in `experiments/*_results.txt`.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q          # 36 tests, ~3 s
cd experiments && OMP_NUM_THREADS=1 python frontier.py   # ~9 min on 2 cores; phase.py ~18 min, phase_high.py ~6, dead_latents.py ~17, rare.py ~4, variants_frontier.py ~7, hierarchy.py ~4 per regime
```

```python
import numpy as np
from saelab import SAE, fit, make_world, feature_recovery, fvu, l0

world = make_world(n_features=64, d=32, l0=3, seed=0)      # 64 true features squeezed into 32 dims
sae = SAE(d=32, n_latents=128, mode="batchtopk", k=3)
fit(sae, world, steps=3000)                                # fresh samples every step
x, a = world.sample(4096, np.random.default_rng(1))
z = sae.encode(x)
print(feature_recovery(world.directions, sae.params["W_dec"]), fvu(x, sae.decode(z)), l0(z))
```

## What is in the box

| Piece | Notes |
|---|---|
| `make_world` / `FeatureWorld` | `n_features` unit directions in `d` dims (random, or orthonormal), each firing independently with probability `p_i` (uniform or Zipf-skewed, expected L0 held fixed) and magnitude `U(0.5, 1.5)`; `x = sum a_i f_i` |
| `SAE(mode="relu", l1=...)` | `a = relu((x - b_dec) W_enc + b_enc)`, loss = MSE + `l1 * sum a` (decoder rows are unit-norm, so the penalty cannot be dodged by rescaling) |
| `SAE(mode="topk", k=...)` | keep each sample's `k` largest activations |
| `SAE(mode="batchtopk", k=...)` | keep the `k * batch` largest activations over the whole batch, so a sample may use 0 or 12 latents; inference uses a threshold tracked as an EMA of the smallest kept activation |
| `JumpReLUSAE` (v0.2) | `a = pre * [pre > theta]`, `theta` learned per latent; loss = MSE + `l0_coef` * mean L0. `theta` trains through a rectangle-kernel straight-through estimator (bandwidth `eps`); the L0 term does not train the encoder. Values above the threshold are untouched, so there is no L1-style shrinkage |
| `GatedSAE` (v0.2) | `f = [gate_pre > 0] * relu(mag_pre)`, `W_mag = W_gate * exp(r_mag)`; loss = MSE + `l1` * sum relu(gate_pre) + an auxiliary reconstruction from the gate alone with the decoder held constant (stop-gradient, implemented by hand) |
| `make_hierarchical_world` / `absorption` (v0.2) | every child feature implies its parent (child fires only when the parent does); `absorption` asks whether a parent's latent goes silent exactly when one of its children is present |
| `fit` | Adam + cosine LR, streaming data, `b_dec` initialised to the data mean, decoder rows renormalised (scale folded into the encoder, function preserved) with the parallel gradient component removed; optional dead-latent resampling toward high-error residuals |
| metrics | `feature_recovery` (best cosine per true feature, and greedy **one-to-one** matching so one latent cannot claim two features), `latent_purity` (what fraction of alive latents are real features), `fvu`, `l0`, `dead_latents`, `coefficient_ratio` (shrinkage: learned activation / true coefficient), `feature_recall` |

Tests: finite-difference gradients for every mode (JumpReLU: all parameters except `theta`, whose straight-through gradient is checked against a loop reference; Gated: checked against the stop-gradient loss, plus a test that the auxiliary term never reaches the decoder; a deliberately wrong JumpReLU gradient is caught) (and a check that a deliberately wrong gradient is caught), TopK has <= k active, BatchTopK averages exactly k in training and its inference mask equals the stored threshold, normalising the decoder preserves the function, projected gradients are orthogonal to the rows, metrics behave on known cases (duplicated decoder rows count once), training is deterministic, and every mode recovers an orthogonal toy world end to end.

Setup for every study: d=32, `fit(steps=3000, batch=512, lr=2e-3)`, 3-5 seeds (a seed changes the world, the init and the data stream), metrics on 4-8k fresh samples. "matched" = fraction of true features with a distinct latent at cosine >= 0.9.

## Study 1: sparsity / fidelity frontier (`frontier.py`, 5 seeds)

64 true features in 32 dims (coherence ~0.55), true L0 = 3, 128 latents. Mean over 5 seeds (std in the results file):

| method | L0 | FVU | matched | purity | recall | ratio |
|---|---|---|---|---|---|---|
| relu l1=0.03 | 38.5 | 0.015 | 0.72 | 0.36 | 0.98 | 0.56 |
| relu l1=0.2 | 9.5 | 0.077 | 1.00 | 0.50 | 0.96 | 0.66 |
| relu l1=0.8 | 4.4 | 0.229 | 1.00 | 0.50 | 0.92 | 0.47 |
| topk k=3 | 2.9 | 0.155 | 1.00 | 0.78 | 0.76 | 0.92 |
| topk k=6 | 5.9 | 0.077 | 1.00 | 0.59 | 0.92 | 0.82 |
| batchtopk k=3 | 3.0 | **0.083** | 1.00 | **0.93** | 0.89 | 0.90 |
| batchtopk k=4 | 4.0 | 0.075 | 1.00 | 0.76 | 0.93 | 0.83 |

(*purity* = alive latents matching a true feature; *recall* = of the events where a recovered feature truly fires, how often its latent does; *ratio* = learned activation / true magnitude, 1.0 is faithful.)

* **Every method finds the features** once it is sparse enough (L1 needs `l1 >= 0.1`; at 0.03 the code is dense, L0 38, and only 72% are matched).
* **TopK at k = true L0 pays for variable sparsity**: the true L0 is *on average* 3, but samples have 0-8 active features, so a fixed k=3 truncates the busy ones: recall 0.76, FVU 0.155. **BatchTopK at the same L0 halves FVU (0.083)** and raises recall to 0.89. It is a mechanism, not under-training: in an exploratory run, TopK's 0.157 FVU barely moved between 3,000 and 10,000 steps (0.156).
* **L1 is expensive**: to reach the FVU of BatchTopK k=4 (0.075, L0 4) it needs L0 ~9.5, 2.4x as many active latents, and its activations are **shrunk** to ~0.65 of the true magnitude (BatchTopK 0.83-0.90). Only half of its (all-alive) latents align with a true feature at cos >= 0.9; I did not investigate what the other half are.
* **BatchTopK finds the number of features**: with k=3 it leaves 59 of 128 latents dead, i.e. ~69 alive for 64 true features, purity 0.93. TopK leaves 46 dead and 0.78 purity.

## Study 2: how much superposition can an SAE undo? (`phase.py`, `phase_high.py`, 3 seeds)

Latents = 2 x features, k = true L0.

* **Sparse regime (true L0 <= 6, up to 8x more features than dimensions, i.e. 256 features in 32 dims): 100% of features recovered by TopK and BatchTopK in all 15 cells.** FVU is ~0 for BatchTopK when features <= dims (a linear encoder can invert the code), and 0.03-0.29 once features outnumber dims. At 8x, BatchTopK is *worse* than TopK on FVU for L0 3 and 6 (0.28/0.29 vs 0.21/0.22) but better at L0 = 1.
* **Dense regime breaks it** (BatchTopK):

| features / d | true L0 | matched | mean best cos | FVU |
|---|---|---|---|---|
| 2 | 12 | 0.20 | 0.84 | 0.097 |
| 2 | 24 | 0.04 | 0.76 | 0.010 |
| 4 | 12 | 0.43 | 0.89 | 0.154 |
| 4 | 24 | 0.25 | 0.81 | 0.011 |
| 8 | 12 | 0.43 | 0.87 | 0.159 |
| 8 | 24 | 0.70 | 0.89 | 0.016 |

  Reconstruction stays good (FVU 0.01-0.16) while feature recovery collapses: **an SAE can reconstruct well and still not have found the features.** Recovery is not monotone in features / d (the 8x, L0=24 cell recovers 70%, the 2x one 4%); each feature fires on 9% of samples at 8x versus 38% at 2x. I have not tested why, so treat "rarer features are easier to separate" as a hypothesis, not a result.

## Study 3: dead latents and resampling (`dead_latents.py`, 4 seeds)

64 features, k = 3, resampling every 300 steps for the first 60% of training. **Null result: resampling changes nothing** (FVU identical to 3 decimals, matched 1.00 either way). With 128 latents, 47 TopK / 59 BatchTopK latents are dead with or without it. The dead latents are surplus capacity, not lost signal: there are only 64 features to find, so resampled latents have nothing to learn and die again. FVU does not depend on the number of latents at all (TopK 0.155, BatchTopK 0.082 from 64 to 512). What does change is the purity of the alive set (TopK 0.79 -> 0.41, BatchTopK 0.92 -> 0.66 going from 128 to 512 latents, no resampling), i.e. extra live latents are duplicates. Resampling would matter when features outnumber latents; I did not test that regime.

## Study 4: do rare features survive? (`rare.py`, 5 seeds)

Zipf(1)-skewed firing, 64 features (the rarest 16 fire in at most 1.3% of samples). Recovery of the 16 most frequent / 16 rarest features: ReLU l1=0.1: 1.00 / 0.99, l1=0.4: 1.00 / 1.00, TopK: 0.98 / 1.00, BatchTopK: 1.00 / 0.99. **No frequency penalty at this scale.** The skew lowers FVU for every method (BatchTopK 0.041 vs 0.083 uniform) because most of the mass sits in a few easy features.

## Study 5 (v0.2): JumpReLU and Gated vs BatchTopK (`variants_frontier.py`, 5 seeds)

Same world as Study 1 (64 true features, d=32, true L0 = 3, 128 latents). `l0_coef` / `l1` swept per architecture; mean over 5 seeds (std in `variants_frontier_results.txt`).

| method | L0 | FVU | matched | purity | dead | ratio | recall |
|---|---|---|---|---|---|---|---|
| jumprelu l0_coef=0.01 | 9.0 | 0.053 | 0.96 | 0.65 | 33 | 0.74 | 0.96 |
| jumprelu l0_coef=0.02 | 4.3 | 0.074 | 0.99 | 0.73 | 41 | 0.82 | 0.93 |
| jumprelu l0_coef=0.05 | 3.1 | 0.084 | 1.00 | 0.77 | 45 | 0.87 | 0.89 |
| jumprelu l0_coef=0.1 | 2.9 | 0.089 | 1.00 | 0.78 | 45 | 0.90 | 0.87 |
| gated l1=0.4 | 7.5 | 0.065 | 0.97 | 0.48 | 0 | 0.73 | 0.95 |
| gated l1=0.8 | 4.8 | 0.083 | 0.98 | 0.49 | 0 | 0.79 | 0.92 |
| gated l1=1.6 | 2.7 | 0.113 | 1.00 | 0.50 | 0 | 0.91 | 0.81 |
| relu l1=0.4 | 6.0 | 0.118 | 1.00 | 0.50 | 0 | 0.61 | 0.95 |
| topk k=3 | 2.9 | 0.155 | 1.00 | 0.78 | 46 | 0.92 | 0.76 |
| batchtopk k=3 | 3.0 | 0.083 | 1.00 | **0.93** | 59 | 0.90 | 0.89 |
| batchtopk k=4 | 4.0 | 0.075 | 1.00 | 0.76 | 43 | 0.83 | 0.93 |

* **JumpReLU ties BatchTopK on the reconstruction/sparsity frontier**: FVU 0.084 at L0 3.1 versus 0.083 at 3.0, and 0.074 at L0 4.3 versus 0.075 at 4.0 (differences far inside one std), with the same recall. It gets there with *no* batch-level top-k: the threshold is learned, and L0 is steered by one coefficient rather than set exactly.
* **But BatchTopK is cleaner**: purity 0.93 versus 0.77 at L0 ~3 (JumpReLU keeps ~14 more alive latents that are not true features), and its sparsity is exact (3.0) where JumpReLU's L0 moves with the coefficient and with the seed.
* **Gated clearly beats L1-ReLU at similar sparsity**: FVU 0.083 at L0 4.8 versus 0.118 at L0 6.0 for ReLU, and much less shrinkage at matched L0 (ratio 0.79 at L0 4.8 against Study 1's 0.47 for ReLU l1=0.8 at L0 4.4). It does *not* remove shrinkage (0.73 at L0 7.5) and it inherits ReLU's problem of every latent staying alive with purity ~0.5.
* Threshold gradient: the encoder never sees the L0 penalty, only the reconstruction error through the active set, and `theta` is updated by the straight-through kernel. A larger `l0_coef` pushed median thresholds up in a test, and the L0 it ends at (2.9-9 across 0.1-0.01) is a smooth, monotone function of the coefficient.

## Study 6 (v0.2): feature absorption in a hierarchy (`hierarchy.py`, 5 seeds)

8 parents (fire w.p. 0.25) x 3 children each (fire w.p. 0.5 *given* the parent, never alone) = 32 features in d=16 (2x superposition), 64 latents, expected true L0 = 5. A child implies its parent, so an SAE under sparsity pressure can fold the parent direction into the child latent and silence the parent latent whenever a child is present (*absorption*). `absorption = 1 - P(parent latent fires | parent + child) / P(parent latent fires | parent, no child)`, over parents with a latent at cosine >= 0.7. Two magnitude regimes: random `U(0.5, 1.5)`, and fixed 1.0 (folding the parent into the child then costs no reconstruction).

| method | L0 | FVU | absorption (random) | absorption (fixed) | P(fires \| no child) | P(fires \| child) |
|---|---|---|---|---|---|---|
| relu l1=0.1 | 18 | 0.021 | 0.11 | 0.13 | 0.92 | 0.82 |
| relu l1=0.4 | 10 | 0.087 | 0.25 | 0.32 | 0.85 | 0.64 |
| jumprelu 0.03 | 7.8 | 0.033 | 0.12 | 0.16 | 0.89 | 0.79 |
| jumprelu 0.1 | 4.5 | 0.074 | 0.28 | 0.38 | 0.79 | 0.60 |
| gated l1=0.4 | 10.8 | 0.028 | 0.12 | 0.14 | 0.87 | 0.78 |
| gated l1=1.6 | 3.9 | 0.122 | 0.32 | 0.37 | 0.58 | 0.44 |
| topk k=5 | 5.0 | 0.095 | 0.45 | 0.57 | 0.69 | 0.39 |
| batchtopk k=5 | 5.0 | 0.058 | 0.26 | 0.36 | 0.82 | 0.64 |

(The last two columns are the random-magnitude regime. Std and the fixed-regime rates are in `hierarchy_*_results.txt`.)

* **Sparsity pressure is the main driver of absorption, with one architecture effect on top.** JumpReLU and Gated sit at 0.12-0.16 at L0 7.8-10.8 and rise to 0.28-0.38 at L0 4-5, where BatchTopK, JumpReLU and Gated are within about one std of each other. **ReLU + L1 is the exception**: at L0 10 it already absorbs 0.25 (random) / 0.32 (fixed), about double Gated at L0 10.8 (0.12 / 0.14; std ~0.04-0.06). My guess, untested, is that an L1 penalty on magnitudes rewards merging a parent into its child's latent more than an L0 or gate-based penalty does.
* **TopK has the most absorption (0.45 / 0.57), but this is confounded**: samples with children are the busy ones, and a hard per-sample k truncates exactly them, so its parent latent is also silenced by truncation (recall 0.52 against 0.70 for BatchTopK). I did not separate truncation from absorption.
* **Fixed magnitudes raise absorption** for most methods, by 0.02-0.12 (e.g. BatchTopK 0.26 -> 0.36, TopK 0.45 -> 0.57), which is the predicted direction, but several of these gaps are about one std with 5 seeds.
* **Recovery is hard here for everyone**: one-to-one matching at cos >= 0.9 reaches only 0.4-0.7 in this world. I have no non-hierarchical control at the same d, so I cannot say how much of that is the hierarchy and how much is the 2x superposition at 64 latents.
* **Limitation of the metric (child_cos = 0.5, `hierarchy_cos_results.txt`)**: when a child direction shares cosine 0.5 with its parent, the metric reads **0.000 absorption for every method** and the parent latent fires *more* often with a child (0.97 vs 0.85-0.9), because the child's activation leaks into the parent's encoder. So "the parent latent fired" cannot detect absorption when features are correlated; a better test would compare the parent latent's activation to the true parent coefficient. Treat the child_cos = 0 numbers as the only valid ones here.

## Caveats

* Synthetic features: independent firing, non-negative coefficients, random directions, one dictionary. Real models have correlated, hierarchical and composed features, so the *rankings* (BatchTopK vs TopK vs L1) are suggestive, not transferable numbers.
* 3-5 seeds per cell; differences within about one std are not claims. The 0.9 cosine threshold is fixed, not tuned.
* The L1 recipe is bare (swept coefficient only; no warm-up or resampling); a tuned L1 baseline might narrow the gaps.
* The `*_results.txt` files hold the outputs of the study scripts; after the sources were re-created, a seed-0 TopK, BatchTopK and L1 run was repeated and reproduced the original FVU and recovery to four decimals.
