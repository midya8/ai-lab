# gaussian-processes

Gaussian processes in NumPy, no SciPy. It started as exact GP regression with analytic gradients for the marginal likelihood and grew from there: sparse approximations (VFE, FITC, DTC, SVGP), random Fourier and Nystrom features, Laplace classification, Student-t and heteroscedastic regression, a spectral mixture kernel, and a Kalman-filter version of Matern GPs that runs in O(n).

Mostly I wanted to see where the cheap approximations are fine and where they quietly mislead you.

```bash
cd gaussian-processes
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q             # ~100 tests
PYTHONPATH=src python -m gaussproc sparse      # also: hyper, features, calibration, extrapolation, scaling
python make_results.py                         # rewrites results/, ~1 min
```

```python
from gaussproc import RBF, ExactGP, SparseGP, greedy_inducing, sample_gp_data

X, y, _ = sample_gp_data(RBF(0.8, 1.5), 400, 0.05, seed=0)
gp = ExactGP(RBF(1.0, 1.0), 0.5).fit(X, y, restarts=3)
mean, var = gp.predict(X[:10], include_noise=True)

Z = X[greedy_inducing(gp.kernel, X, 20)]
sp = SparseGP(RBF(1.0, 1.0), Z, 0.5, "vfe").fit(X, y)
print(sp.bound(), "<=", gp.log_marginal_likelihood())
```

## Things I found

**Signal variance basically can't be learned.** Noise and lengthscale tighten as n grows, but the amplitude estimate at n=300 still ranges 0.72-1.91 (truth 1.5). A draw on a finite interval only has a handful of independent bumps.

**Sparse GPs are fine once you have enough inducing points.** On a 1500-point 1-D problem, VFE, FITC and DTC all match the exact GP from about 24 inducing points (RMSE 0.049). Below that VFE explains what it can't fit as extra noise, while FITC shrinks the lengthscale and underestimates noise. At 2000 points VFE with 50 inducing points is about 440x faster than the exact GP.

**Random features are overconfident.** Nystrom is essentially exact by 50 features on a smooth 2-D kernel. Random Fourier features still have 10% kernel error at 500, and their predictive variance is too small (31% of the true variance at 10 features, 91% at 500).

**Kernel choice barely affects calibration.** 95% intervals cover 93.5-93.7% whichever Matern/RBF family you pick, even the wrong one. Picking the right family from 80 points with the marginal likelihood works for RBF (75%) and is worse than chance for Matern 5/2.

**Structure wins at extrapolation.** On a trend plus a periodic signal, the periodic + linear kernel finds the true period from the likelihood alone and extrapolates with RMSE 0.14. A flexible Matern + linear kernel gets 2.17.

Raw numbers live in `results/*.json`.

## Limitations

Synthetic data throughout, constant mean, sparse hyperparameters fit with finite-difference gradients and fixed inducing locations, and the exact GP is meant for a few thousand points at most.
