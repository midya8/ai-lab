# flowlab

**Normalizing flows from scratch in NumPy**: RealNVP affine couplings, NICE additive couplings, ActNorm with data-dependent init and fixed permutations — every layer with a **hand-written backward pass checked against central finite differences** — evaluated on 2-D densities whose *exact* log-density is known, so the quality of a flow is measured as a KL gap to the truth instead of a number with no reference.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                        # 25 tests, ~4 s
cd experiments
OMP_NUM_THREADS=1 python depth_study.py ring8             # ~6 min on 2 cores
OMP_NUM_THREADS=1 python depth_study.py checker4          # ~6 min
OMP_NUM_THREADS=1 python nll_study.py; OMP_NUM_THREADS=1 python gallery.py
```

```python
from flowlab import Flow, fit, gaussian_ring, nll_gap, bridge_mass, ascii_density

ds = gaussian_ring()                       # 8 Gaussians on a circle, exact density
f = Flow(d=2, n_layers=8, hidden=64, seed=0)
fit(f, ds.sample(20000, 0), steps=3000)
print(nll_gap(f, ds))                      # {'flow_nll': ..., 'true_nll': ..., 'gap': ...}  gap = KL(data || flow) in nats
print(bridge_mass(f.sample(20000), ds.centers, ds.sigma))   # mass the flow puts between the modes
print(ascii_density(f.sample(20000), lim=3.2))
```

## What is in the box

| Piece | Notes |
|---|---|
| `Coupling` | `y_b = x_b·exp(s) + t`, `[s, t] = MLP(x_a)`, `s = 2·tanh(raw)` (bounded); `additive=True` fixes `s = 0` (volume preserving, logdet 0). Exact inverse, exact logdet, hand-written backward. Last MLP layer is zero-initialised so a fresh flow is the identity |
| `ActNorm` | per-coordinate scale + shift, initialised from a data batch so activations start zero-mean/unit-variance |
| `Permutation` | fixed coordinate shuffle between couplings (reversal for d=2) |
| `Flow` | stack onto a standard-normal base; `log_prob`, `nll`, `loss_and_grads`, `inverse`, `sample(temperature=)`; any `d >= 2` |
| `fit` | Adam, cosine LR, global-norm clipping, held-out NLL tracking |
| datasets | `gaussian_ring` and `checkerboard` have exact densities; `two_moons`, `spiral` do not |
| metrics | `nll_gap` (flow NLL − exact entropy on held-out samples ≈ KL in nats), `bridge_mass` (probability mass between modes), `density_grid_integral` (does exp(log_prob) integrate to 1?) |
| `ascii_density` | dependency-free log-scaled terminal plot |

Tests (`tests/test_flowlab.py`): inverse round-trip (affine and additive, d=2 and 5); `logdet` against `log|det J|` of a numerically differentiated Jacobian; additive layers have logdet exactly 0; **every parameter's gradient against finite differences** (affine/additive, d=2/5) — and a test that *breaks* the logdet term in the backward pass and asserts the check catches it; zero-init is the identity; ActNorm standardises; an untrained random flow still integrates to ≈1 on a grid; ActNorm alone recovers a Gaussian's closed-form entropy; training beats the Gaussian baseline; `sample` is the exact inverse of the base noise; the ring and checkerboard densities integrate to 1 and their Monte-Carlo entropy matches a grid entropy.

## Studies

All runs: 3 seeds (each seed changes the training set and the initialisation), 20,000 training points, 3,000 Adam steps of batch 256, hidden width 64, held-out set of 20,000 fresh points. Raw output is in `experiments/*_results.txt`. "gap" is flow NLL minus the data's exact NLL in nats per example, i.e. a Monte-Carlo estimate of KL(data ‖ flow); 0 is perfect.

### 1. Depth and coupling type on the 8-Gaussian ring (`depth_study.py ring8`)

A single connected unimodal base distribution has to be bent into 8 separate blobs, so the flow must put some mass *between* the modes. `bridge%` is the share of samples more than 3.5σ from every mode (an exact sampler gives 0.2%).

| coupling | L | params | gap (nats) | bridge % |
|---|---|---|---|---|
| affine | 2 | 520 | 1.202±0.054 | 39.7±2.8 |
| affine | 4 | 1036 | 0.597±0.031 | 13.4±0.3 |
| affine | 8 | 2068 | 0.495±0.020 | 11.0±0.8 |
| affine | 16 | 4132 | 0.400±0.036 | 8.1±0.9 |
| additive | 2 | 390 | 1.541±0.001 | 50.7±0.2 |
| additive | 4 | 776 | 1.121±0.056 | 37.3±0.6 |
| additive | 8 | 1548 | 0.642±0.064 | 18.4±3.2 |
| additive | 16 | 3092 | 0.431±0.068 | 10.6±1.4 |

* Depth helps monotonically for both kinds, and most of the bridge mass is a *topology* cost: even 16 affine couplings still leave 8% of samples between modes (40× the exact rate).
* The extra expressiveness of the scale term matters at equal depth (L=2: gap 1.20 vs 1.54), but the volume-preserving flow closes in: at L=16 the gap difference (0.400 vs 0.431) is within about one seed std. Additive layers have ~25% fewer parameters at equal depth, but compare by parameters, not depth: additive L=16 (3,092 params, gap 0.431) beats affine L=8 (2,068 params, 0.495) here, using 50% more parameters.

### 2. Depth and coupling type on the checkerboard (`depth_study.py checker4`)

Uniform on 8 black squares with hard edges: no flow can represent the discontinuities exactly, so the gap never reaches 0.

| coupling | L | params | gap (nats) |
|---|---|---|---|
| affine | 2 | 520 | 0.587±0.002 |
| affine | 4 | 1036 | 0.476±0.017 |
| affine | 8 | 2068 | 0.380±0.010 |
| affine | 16 | 4132 | 0.316±0.010 |
| additive | 2 | 390 | 0.702±0.001 |
| additive | 4 | 776 | 0.612±0.010 |
| additive | 8 | 1548 | 0.523±0.018 |
| additive | 16 | 3092 | 0.473±0.004 |

Here the scale term clearly matters: affine L=8 (0.380) beats additive L=16 (0.473) with 33% fewer parameters. The two datasets disagree about whether volume preservation is a real handicap — on the ring it is a depth tax, on sharp-edged uniform density it is a hard floor within this budget. Both are 3-seed means at a fixed training budget, not converged optima.

### 3. Datasets without a closed form (`nll_study.py`)

Held-out NLL (nats/example, lower is better), affine couplings. The reference is a diagonal Gaussian (ActNorm alone).

| dataset | diag Gaussian | L=2 | L=4 | L=8 | L=16 |
|---|---|---|---|---|---|
| moons | 2.016 | 1.426±0.092 | 1.213±0.025 | 1.089±0.025 | 1.053±0.021 |
| spiral | 3.566 | 2.702±0.097 | 1.807±0.013 | 1.450±0.119 | 1.244±0.079 |

The spiral keeps improving to L=16 (still not saturated); moons flattens after L=8. With no exact density there is no gap to report, so these numbers are only comparable with each other.

### Gallery (`gallery.py`): 8 affine couplings, data left, samples right

Run `python gallery.py` for the ring and checkerboard figures (stored in `experiments/gallery_results.txt`). At this budget the flow's samples are visibly worse than the data: the ring's eight blobs are joined by smeared bridges and the checkerboard's squares are blurred into each other — consistent with the 0.50 / 0.38 nat gaps above.

## Honest limits

* 2-D toy data only; no claim about image flows. The hidden MLP is tiny (64 units) and each run is 3,000 steps, so every gap is an upper bound on what the architecture can do.
* The gap is a Monte-Carlo estimate on 20,000 held-out points; it is finite on the checkerboard only because the flow assigns non-zero density outside the squares.
* Seeds change data and initialisation together; three seeds give a rough spread, not a confidence interval.
* No spline couplings, no autoregressive flows, no continuous-time flows yet (see the roadmap).

## v0.2: learned invertible linear mixing (Glow-style)

`InvLinear` (`Flow(..., mixing="linear")`) replaces each fixed permutation with a learned d x d matrix
(orthogonal init, logdet = log|det W|, gradient `x^T gy + sum(glogdet) W^-T`, finite-difference checked).
`mixing_study.py` compares it with permutations at equal depth (6 couplings, hidden 48, 1500 steps, 3 seeds,
held-out NLL in nats, results in `results/mixing.txt`):

| dataset | permutation | linear |
|---|---|---|
| moons | 1.165 ± 0.029 | 1.152 ± 0.026 |
| spiral | 1.831 ± 0.064 | 1.998 ± 0.113 |
| checker4 | 2.554 ± 0.019 | 2.573 ± 0.041 |

Result: no consistent benefit in 2-D (within noise on moons/checker4, worse on spiral); with d=2 a permutation
is already enough to mix coordinates, so Glow's advantage is expected only in higher dimensions.
