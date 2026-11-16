# diffusionlab

A diffusion model (DDPM / DDIM / classifier-free guidance) written from scratch in NumPy — hand-written backprop, no autodiff, no framework — on 2-D data whose noised marginals are known **in closed form**. That last part is the point: because the Bayes-optimal noise predictor `eps*(x_t, t)` is exact, you can measure how far a trained network is from the optimum, and you can run the samplers with a *perfect* model to separate discretisation error from learning error.

```
data: 8 Gaussians on a ring (radius 1.5, std 0.3 -> neighbours overlap)
net : conditional MLP (3 x 128, SiLU, sinusoidal time features, class one-hot / null) = 36.7k params
loss: epsilon-prediction MSE, t ~ U{1..100}, linear schedule, Adam + cosine LR + EMA, label dropout 10%
```

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m diffusionlab --steps 4000      # train (~15 s), error-vs-optimum table, sampler table, ASCII scatter
PYTHONPATH=src python -m pytest -q                      # 23 tests (~15 s)
python experiments/run_all.py                           # the tables below (3 seeds, ~4 min)
```

```python
from diffusionlab import ring, linear_schedule, EpsNet, train, sample, net_eps_fn
g, s = ring(), linear_schedule(100)
net = EpsNet(2, 128, 3, g.K, T=100)
train(net, lambda n, r: g.sample(n, r, True), s, steps=4000)
x = sample(net_eps_fn(net, guidance=1.0), s, 1000, 2, steps=50, eta=0.0, labels=[3] * 1000)   # class-3 samples
```

## What is in it

| module | contents |
|---|---|
| `schedule.py` | linear (rescaled so total noise is independent of T) and cosine schedules, true posterior variance |
| `gmm.py` | isotropic Gaussian mixture: exact noised marginal, score, `eps_star`, class-conditional `eps_star_class`, mode statistics |
| `net.py` | `EpsNet` with hand-written backprop (finite-difference gradient-checked, incl. null labels), Adam, EMA |
| `diffusion.py` | training loop, `q_sample`, generalised DDIM sampler (`eta=0` deterministic, `eta=1, steps=T` = DDPM ancestral), classifier-free guidance, exact-score oracle |
| `diagnostics.py` | excess error vs the exact optimum by noise level, ASCII scatter |
| `metrics.py` | sliced Wasserstein-1, unbiased multi-kernel MMD |

Tests check: exact score equals a finite-difference gradient of the log density at four noise levels; forward-process moments match the closed form; `eps*` is the conditional mean (perturbing it raises MSE and the Pythagoras decomposition `loss = excess + irreducible` holds); DDIM at `eta=1` reproduces Ho et al.'s posterior mean and variance exactly; DDIM is deterministic given `x_T`; the oracle sampler recovers the distribution to within the finite-sample floor; one-step sampling collapses toward the mean; guidance arithmetic; metric sanity (sliced-W1 of a unit shift = 2/pi).

## Results (3 seeds, 6000 training steps, 8000 samples; raw output in `experiments/results.txt`)

**1. How far is the trained net from the optimum?** `excess = E|eps_theta - eps*|^2` (per coordinate, unconditional prediction), against `irreducible = E|eps* - eps|^2`, which even a perfect model pays:

| t | abar | excess (mean, min..max over seeds) | irreducible |
|--:|--:|--:|--:|
| 1 | 0.999 | 0.0084 (0.0072..0.0104) | 0.994 |
| 12 | 0.864 | **0.0206** (0.0188..0.0219) | 0.690 |
| 23 | 0.583 | 0.0019 | 0.572 |
| 45 | 0.123 | 0.0007 | 0.145 |
| 67 | 0.009 | 0.0005 | 0.011 |
| 100 | 0.000 | 0.0010 (0.0005..0.0018) | 0.000 |

In absolute terms the net's excess error is largest at low noise (t ~ 12, 0.021), where the score has to resolve the 0.3-wide modes. Relative to the irreducible error it is under 1% at t = 1, 23, 34 and 45, ~1% at t = 56, ~3% at t = 12 and ~5% at t = 67; beyond t ~ 78 the irreducible error vanishes (x_t is nearly pure noise, so eps is almost determined by x_t) and the net's small excess (~0.0005) is the whole error. The training loss alone (~0.13) cannot show any of this; it is dominated by the irreducible part.

**2. Sampler quality** (sliced-W1 to 8000 fresh true samples; the finite-sample floor between two exact sets is 0.0274 +- 0.0063; "oracle" = same sampler with the exact `eps*`):

| steps | eta | net sW1 | net on-mode | oracle sW1 | oracle on-mode |
|--:|--:|--:|--:|--:|--:|
| 100 | 1 (DDPM) | 0.0452 | 98.0% | 0.0292 | 99.8% |
| 100 | 0 | 0.0406 | 97.4% | 0.0325 | 99.6% |
| 50 | 0 | 0.0453 | 97.4% | 0.0374 | 99.6% |
| 20 | 0 | 0.0660 | 97.1% | 0.0557 | 99.7% |
| 10 | 0 | 0.1026 | 96.2% | 0.0909 | 99.3% |
| 5 | 0 | 0.1874 | 93.8% | 0.2157 | 94.1% |
| 20 | 1 | 0.0842 | 97.4% | 0.0633 | 99.7% |
| 10 | 1 | 0.1386 | 95.3% | 0.1131 | 99.2% |

Reading: with the exact model the DDPM sampler sits at the floor (0.0292 vs 0.0274); the trained net adds ~0.008-0.016 on top at 10-100 steps, visible as a 2-3 point drop in on-mode rate (on-mode = within 3 std of a mean; the true rate is 99.8%). Below ~20 steps discretisation error dominates and the gap between net and oracle shrinks. Deterministic DDIM beats stochastic sampling at every reduced step count (20 steps: 0.066 vs 0.084), as expected for a coarse grid. At 5 steps the net scores *better* sW1 than the oracle (0.187 vs 0.216) while on-mode rates are equal; I would not read that as the net being "better" — a single metric, three seeds, a regime where both samplers are bad.

**3. Classifier-free guidance** (50-step DDIM, 8 classes x 1000 samples per seed; accuracy = nearest mode is the requested class; spread = mean per-axis std of the samples; true within-class std 0.3; oracle column uses seed 0's exact conditional/unconditional `eps*` only):

| scale | net acc | net spread | oracle acc | oracle spread |
|--:|--:|--:|--:|--:|
| 0 (uncond.) | 0.125 | 1.084 | 0.116 | 1.081 |
| 1 (cond.) | 0.963 | 0.279 | 0.966 | 0.274 |
| 2 | 1.000 | 0.160 | 1.000 | 0.160 |
| 4 | 1.000 | 0.157 | 1.000 | 0.111 |
| 8 | 1.000 | 0.250 | 1.000 | 0.086 |

Reading: unguided class-conditional accuracy 0.963 matches the exact model's 0.966 — that is the ceiling, since neighbouring modes genuinely overlap. Guidance buys accuracy (to 1.0 at scale 2) by shrinking diversity: the exact guided sampler's spread collapses from 0.27 to 0.09 as the scale grows (mode-seeking). The trained net tracks the oracle up to scale 2 but at scales 4-8 it stops shrinking (0.157 -> 0.250 spread), i.e. extrapolating `eps_u + w(eps_c - eps_u)` far beyond the training distribution amplifies the net's errors — the guided field leaves the region the net saw. Accuracy is saturated so it cannot show this; the spread does.

## Honest limits

- 2-D toy data with an analytically tractable score. This demonstrates the *machinery and its error budget*, not image-generation quality; the conclusions (e.g. where the net's error sits) are specific to this data and size.
- Training data are drawn fresh from the mixture every step (an effectively infinite dataset), so there is no overfitting/memorisation study here.
- 3 seeds; ranges are shown for table 1, but tables 2 and 3 are means only. Differences of a few thousandths in sW1 are within the sampling noise (floor std 0.006).
- The oracle in table 3 uses a single seed's samples (the oracle has no training seed; only the sampling noise varies).
- The training schedule (steps, lr, EMA decay 0.995) was set once from a 3000-step smoke run and then fixed; no per-result tuning.
- float64 NumPy, small MLP: fast enough for 4000 steps in ~15 s, not a framework.

## Spiral study (non-analytic data, v0.2)

`shapes.spiral` is a two-arm spiral with no closed-form score, so only sample metrics apply (`experiments/spiral.py`, output in `experiments/spiral_results.txt`; single seed, 4000 steps). Floor between two exact sets: sliced-W1 0.0162 +- 0.0027; pure noise: 0.1563. Deterministic DDIM (eta=0) beat ancestral (eta=1) at every step count: sliced-W1 0.2160 / 0.0933 / 0.0436 / 0.0288 / 0.0247 at 5 / 10 / 25 / 50 / 100 steps. Even at 100 steps the net stays above the floor. Single seed, so differences of a few thousandths are not established.

## DPM-Solver-2 (v0.2)

`diffusionlab.dpm.sample_dpm2` is a midpoint second-order solver in half-log-SNR
(2 eps-evaluations per step). Sliced-Wasserstein to ring data, oracle eps, linear T=100,
4000 samples, one seed, same starting noise (equal NFE = evaluations):

| NFE | DDIM | DPM-Solver-2 |
|----:|-----:|-------------:|
| 4 | 0.302 | 0.696 |
| 6 | 0.160 | 0.366 |
| 8 | 0.106 | 0.202 |
| 12 | 0.070 | 0.072 |
| 16 | 0.055 | 0.039 |
| 24 | 0.039 | 0.031 |

Honest reading: with this coarse 100-step discrete schedule, the second-order solver is
**worse** below ~12 NFE (few, large steps; the midpoint timestep is rounded to an integer)
and wins from 16 NFE up. Single seed; the test only asserts the NFE=16 win.

## v-prediction and SNR weights (`diffusionlab.vparam`)

Exact conversions between eps, x0 and v parametrisations (`v = alpha*eps - sigma*x0`), wrappers that expose any eps-predictor as a v-predictor and back, and per-timestep loss multipliers (eps, x0, v, min-SNR-gamma). Tests verify round-trips and that a v-wrapped oracle reproduces DDIM / DPM-Solver-2 samples. These are utilities only; training with a v or min-SNR loss is not yet benchmarked here.
