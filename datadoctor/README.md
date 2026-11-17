# datadoctor

Finds the rows in a labelled tabular dataset that are probably mislabelled, duplicated, leaking from train into test, or outliers. The label-error part is confident learning (Northcutt et al., 2021) written from the paper's definitions, using out-of-fold probabilities from whatever model you plug in. NumPy only.

```bash
cd datadoctor
PYTHONPATH=src python -m datadoctor audit data.csv --label species --text description --out audit.md
PYTHONPATH=src python -m datadoctor audit data.csv --label species --fail-above 0.05    # CI gate
PYTHONPATH=src python -m datadoctor demo --noise 0.3 --kind pairwise
PYTHONPATH=src python experiments/benchmark.py
PYTHONPATH=src python -m pytest -q
```

Flagged rows are candidates for a human to look at, not verdicts. The report (Markdown, JSON or a single HTML file) shows class balance, the estimated noise matrix, flagged rows with a suggested label, duplicate groups (including exact duplicates that disagree on the label) and per-class outliers.

## What I found

On synthetic data with injected noise (full table in `experiments/results.md`):

- **It pays off on structured noise.** With 30% pairwise label confusion, a model trained on the noisy labels drops to 0.72 test accuracy. Pruning the flagged rows gets it back to 0.82 with softmax probabilities, or 0.87 with kNN (clean labels would give 0.895). With random noise, pruning only buys 0.5-1.5 points.
- **The model has to fit the data.** On a dataset a linear model can't separate, linear probabilities make the detector noticeably worse than kNN ones.
- **Simple baselines are strong.** In-sample loss from a regularised linear model ranks errors about as well as cross-validation on easy data.
- **Realistic noise is much harder.** When flips are concentrated near class borders (more like real annotators), precision stays high (0.87-0.96) but recall falls to about 0.25, and the noise rate is underestimated (0.06 vs 0.20). Uniform-noise benchmarks flatter every detector.

## It grew a lot

Regression targets (robust out-of-fold residuals), multi-label issues, audit diffs for CI, relabel suggestions, and a whole set of drift checks: KS/PSI with bootstrap CIs, Wasserstein, MMD, adversarial validation, categorical drift, label shift and BBSE, and covariate-shift weights. Most of these are tested on synthetic data only.
