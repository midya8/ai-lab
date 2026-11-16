# conformlab

Conformal prediction from scratch in NumPy: distribution-free prediction intervals and sets with a finite-sample coverage guarantee, and an honest look at where that guarantee is **not** enough. Everything runs on synthetic data with known ground truth, so coverage is measured against the truth rather than asserted.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q          # 13 tests (~2 s)
python experiments/run_all.py               # regenerates experiments/results.txt (~1 min)
```

```python
from conformlab import *
rng = np.random.default_rng(0)
Xtr, ytr = hetero_regression(2000, rng); Xc, yc = hetero_regression(1000, rng)
feat = RFF(1, 120, 0.7)
lo = QuantileRegressor(feat, 0.05).fit(Xtr, ytr); hi = QuantileRegressor(feat, 0.95).fit(Xtr, ytr)
cqr = CQR(lo.predict, hi.predict, alpha=0.1).calibrate(Xc, yc)
lower, upper = cqr.predict(Xnew)            # >= 90% marginal coverage, any model, any distribution
```

## What is in it

| module | contents |
|---|---|
| `conformal.py` | `conformal_quantile` (the ceil((n+1)(1-a))-th order statistic, +inf if n is too small), `SplitConformal` (constant or normalised width), `CQR`, `ConformalClassifier` (LAC, randomised APS, optional Mondrian per-class calibration), `aci` (adaptive conformal inference for streams), `quantile_tracker` (calibration-free online quantile tracking with a distribution-free long-run miss-rate bound), `MondrianSplitConformal` (group-conditional regression intervals; infinite for under-populated or unseen groups) |
| `models.py` | base learners in NumPy: random-Fourier-feature ridge, pinball-loss quantile regression (Adam), softmax regression |
| `data.py` | heteroscedastic 1-D regression (Gaussian or heavy-tailed t3 noise) and Gaussian classes with two intrinsically hard classes |
| `metrics.py` | coverage, width, equal-count binned coverage, class-conditional coverage, set size, the exact Beta law of coverage |

Tests check: the quantile rank rule; the **exact** Beta(k, n+1-k) law of coverage (mean and variance over 3000 calibration draws); marginal coverage for every method; CQR and Mondrian beating their plain counterparts on conditional coverage; APS score extremes and set/score consistency; ACI's deterministic long-run bound `|miss rate - alpha| <= (max(alpha,1-alpha)+gamma)/(gamma T)` even on a Cauchy stream.

## Results (alpha = 0.1, 5 seeds, test n = 30,000; full tables in `experiments/results.txt`)

**Regression with noise that grows toward the edges.** Every method hits 90% marginal coverage, as promised. The marginal number hides the problem:

| method (Gaussian noise) | coverage | mean width | worst x-bin | best x-bin |
|---|---|---|---|---|
| constant-width | 0.905 | 1.18 | **0.772** | 1.000 |
| normalised by a fitted sigma(x) | 0.902 | 1.00 | 0.887 | 0.918 |
| CQR | 0.900 | 1.00 | 0.880 | 0.921 |

Constant-width intervals are 15% wider *and* under-cover the noisy region by 13 points. Normalised and CQR are tied; with heavy-tailed t3 noise they are also ~10-13% narrower than constant-width. (On x ~ N(0,1) the noise barely varies over the data, so all three tie in width; the conditional gap still shows for constant-width.)

**Classification, 6 classes, two hard ones** (base accuracy 0.725):

| method | coverage | set size | worst class | hard-class coverage |
|---|---|---|---|---|
| LAC | 0.895 | 1.66 | 0.703 | 0.738 |
| APS (randomised) | 0.898 | 1.94 | 0.744 | 0.775 |
| LAC + Mondrian | 0.900 | 2.00 | 0.880 | 0.897 |
| APS + Mondrian | 0.897 | 2.18 | 0.877 | 0.894 |

Marginal 90% is met by all, but plain LAC gives the hard classes ~74%. Mondrian calibration restores per-class ~90% for 0.34 extra labels per set. Randomised APS produced empty sets 1.2% of the time here (3% with Mondrian); LAC never did, and gave the smallest sets. APS did not beat LAC on conditional coverage in this setup, contrary to its reputation, though the setup (well-specified softmax, low dimension) favours LAC.

**Online shift** (residual scale doubles at the midpoint, T = 6000): static split conformal misses 39.9% after the shift. ACI brings the long-run miss rate to 0.100-0.103 for gamma in {0.005, 0.02, 0.05}. The cost is visible in `alpha_t`: it falls to ~0.004, i.e. the method is using near-maximal quantiles (and, at times, infinite intervals) of a 500-point calibration set, because a 2x scale shift can't be matched by a quantile of unshifted scores at a normal level. Coverage is restored by sometimes being uninformative.

**Covariate shift** (`experiments/shift_study.py`; train/calibrate on x~U[-3,3], test on x~N(1.5, 0.7) truncated, 5 seeds): weighted conformal (Tibshirani et al. 2019, `WeightedSplitConformal`) with the density ratio known or estimated by a logistic domain classifier (`logistic_density_ratio`).

| method | coverage | width |
|---|---|---|
| unweighted (ignores shift) | 0.936 | 1.18 |
| weighted, exact ratio | 0.901 | 0.99 |
| weighted, logistic-estimated | 0.910 | 1.03 |

In this setup the shift moves test points into a *quieter* region, so ignoring it over-covers (wastefully wide) rather than under-covers; weighting restores ~0.90 and gives 16% narrower intervals. I did not construct a harmful-direction shift here; the guarantee is the same either way. The estimated ratio is slightly conservative (0.910).

## Caveats

- Marginal guarantees assume exchangeability; the shift experiment is exactly where that fails, and ACI only guarantees the *average* miss rate, not per-window coverage.
- Test-set coverage is itself random: with a 1000-point calibration set the coverage of one fitted interval varies around 0.90 with std ~0.01 (the Beta law the tests check).
- Hyperparameters (RFF width 120, lengthscale 0.7, ridge/steps) were fixed before looking at coverage; the sigma-hat model is a ridge fit to absolute training residuals, a crude choice that still worked.
- All data are synthetic; no real-dataset claim is made.


## RAPS (regularised APS)

`ConformalClassifier(..., method="raps", lam=0.05, k_reg=2)` adds a rank penalty `lam * max(0, rank - k_reg)` to the randomised APS score, discouraging large sets padded with tiny-probability labels. `lam=0` is exactly APS (tested); coverage stays valid for any fixed `lam`/`k_reg`.

Single-seed check on the Gaussian-classes data (30k test points, alpha=0.1; settings tried by hand, not tuned on held-out data, so treat as indicative only):

| method | coverage | mean set size |
|---|---|---|
| LAC | 0.900 | 1.694 |
| APS | 0.892 | 1.904 |
| RAPS lam=0.02,k=1 | 0.891 | 1.845 |
| RAPS lam=0.05,k=2 | 0.893 | 1.850 |
| RAPS lam=0.2,k=2 | 0.893 | 1.793 |

RAPS shrinks APS sets modestly here but stays above LAC in size; the effect is expected to be larger with many classes (this data has 6).


## Jackknife+ / CV+

`JackknifePlus(fit_fn, alpha, folds=None)` uses every point for training and calibration (no held-out split). `fit_fn(X, y)` returns a predict callable; `folds=None` is leave-one-out, `folds=K` is CV+. Guarantee is coverage >= 1-2*alpha; tests check it against a brute-force implementation of the formula and empirically (about 0.9 at alpha=0.1 on a linear-Gaussian toy, which is a synthetic check, not a benchmark).

## Conformal predictive distributions

`ConformalPredictiveDistribution(mean_fn, scale_fn, randomised)` turns calibration residuals into a full predictive CDF `F(y|x)`, quantiles, equal-tailed asymmetric intervals and CRPS. With `randomised=True` the PIT is exactly uniform marginally (checked in `tests/test_cpd.py` on the synthetic heteroscedastic data); a correct scale function gives lower CRPS than a constant one.

## Size-stratified coverage

`size_stratified_coverage(S, y)` reports coverage within each prediction-set-size stratum and `worst_size_stratum(S, y, min_count)` the lowest one among strata with enough points. Marginal coverage can hold while the large-set (hard) inputs are under-covered; this diagnostic exposes it. Tests check a hand example and that the strata recombine to the marginal coverage.

## Set-outcome breakdown and worst class

`set_outcome_breakdown(S, y)` splits test points into empty / correct singleton / wrong singleton / multi-label covered / multi-label missed (fractions sum to 1; correct singleton + multi covered equals coverage, tested). `worst_class_coverage(S, y, K, min_count)` returns the lowest-coverage class among classes with enough points, a one-line check for class-conditional failure.

## Calibration-size planning

`exact_expected_coverage(n, alpha)` gives the exact marginal coverage of split conformal with `n` calibration points (continuous scores): `ceil((n+1)(1-alpha))/(n+1)`. `calibration_size_for_slack(alpha, slack)` returns the smallest `n` whose over-coverage is at most `slack`. A simulation test checks the formula.

## Exchangeability monitor (martingale.py)

`detect_change(scores, delta)` runs randomised online conformal p-values through a simple-mixture power martingale; by Ville's inequality the false-alarm probability is <= delta under exchangeability. Tests: 200 iid streams at delta=0.1 alarm <= 15%; a +2 sigma mean shift at t=300 is caught. Quick check (50 seeds, delta=0.01, shift 0 -> 2 sigma at t=300): 50/50 detected, median delay 32 steps. Synthetic Gaussian data only.

## PAC / training-conditional guarantee

Marginal coverage is an average over calibration sets; one particular calibration set can under-cover. `prob_coverage_below(n, alpha, target)` gives the exact probability that a single split-conformal set has coverage below `target` (coverage ~ Beta(k, n+1-k), evaluated through the binomial identity, no SciPy). `pac_calibration_size(alpha, eps, delta)` returns the smallest `n` with P(coverage < 1-alpha-eps) <= delta. For alpha=0.1: (eps=0.05, delta=0.05) needs n=85, (0.02, 0.05) needs 567, (0.02, 0.01) needs 1284, versus n=9 for the marginal guarantee. Tests compare against a 40k-run simulation and check minimality. Continuous scores assumed.
