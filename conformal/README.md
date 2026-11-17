# conformal

Conformal prediction in NumPy: split conformal, normalised intervals, CQR, LAC/APS/RAPS sets, Mondrian (per-group) calibration, jackknife+/CV+, adaptive conformal for drifting streams, weighted conformal for covariate shift, and conformal predictive distributions. All on synthetic data, so coverage gets checked against the truth.

The thing I wanted to show is how often "90% coverage" is technically true and still misleading.

```bash
cd conformal
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q     # 80 tests
python experiments/run_all.py          # regenerates experiments/results.txt (~1 min)
```

```python
from conformal_pred import *
rng = np.random.default_rng(0)
Xtr, ytr = hetero_regression(2000, rng); Xc, yc = hetero_regression(1000, rng)
feat = RFF(1, 120, 0.7)
lo = QuantileRegressor(feat, 0.05).fit(Xtr, ytr)
hi = QuantileRegressor(feat, 0.95).fit(Xtr, ytr)
cqr = CQR(lo.predict, hi.predict, alpha=0.1).calibrate(Xc, yc)
lower, upper = cqr.predict(Xnew)
```

## What came out of it

Regression with noise that grows toward the edges (alpha 0.1, 5 seeds): all three methods hit 90% overall, but constant-width intervals are 15% wider *and* only cover 77% in the noisiest region. Normalised intervals and CQR get every region to roughly 88-92%.

Classification with 6 classes, two of them hard: plain LAC covers the hard classes about 74% of the time while reporting 90% overall. Mondrian calibration fixes that (~90% per class) for 0.34 extra labels per set. APS didn't beat LAC here, which surprised me.

Online shift where the noise doubles halfway: static split conformal misses 40% afterwards. ACI brings the long-run miss rate back to about 0.10, but partly by handing out huge or infinite intervals for a while.

Covariate shift: weighted conformal with an estimated density ratio gets 0.910 coverage and 16% narrower intervals than ignoring the shift.

One small table I keep coming back to: for a PAC-style guarantee (coverage below 85% with probability at most 5%) you need 85 calibration points, and 567 for a 2-point margin, vs 9 for the marginal guarantee.

## Caveats

Everything assumes exchangeability except the shift experiments, and ACI only controls the average miss rate, not any particular window. All data is synthetic.
