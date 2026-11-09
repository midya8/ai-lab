# datadoctor

Find the rows in a labelled tabular dataset that are probably **mislabelled**, **duplicated** (exactly, nearly, or with conflicting labels), **leaking** from train into test, or **outliers**, from scratch in NumPy. The label-error detector is *confident learning* (Northcutt et al., 2021) written from the paper's definitions, with out-of-sample probabilities from any model you plug in.

```bash
export PYTHONPATH=src
python -m datadoctor audit data.csv --label species --text description --out audit.md
python -m datadoctor audit data.csv --label species --write-clean clean.csv    # drops the flagged rows
python -m datadoctor audit data.csv --label species --fail-above 0.05          # CI gate: exit 1 if >5% flagged
python -m datadoctor demo --noise 0.3 --kind pairwise                          # inject noise, score the detectors
python experiments/benchmark.py                                                # the table below
```

Features are every column that parses as a number; the label can be any string. `--text` adds MinHash/LSH near-duplicate text detection.

## What it does

1. **Out-of-sample probabilities.** Stratified k-fold (each row is scored by a model that never saw it, optionally averaged over repeated splits). Models: a NumPy softmax regression (Adam, L2) and a k-NN classifier; anything with `fit(X, y, n_classes=)` / `predict_proba(X)` works.
2. **Confident joint.** Per-class thresholds t_j = mean self-confidence of class j; C[i][j] counts rows labelled *i* whose probability for *j* clears t_j. Calibrated to the label counts, it yields the **noise matrix** P(given | true), the **true class prior** and an overall **noise-rate estimate**.
3. **Label-issue rules.** `argmax`, `confident_joint` (off-diagonals), `prune_by_class`, `prune_by_noise_rate`, and `both` (their intersection, the default). Plus per-row **quality scores** to rank by review priority: self-confidence, normalised margin, entropy-weighted.
4. **Duplicates.** Exact groups (and the ones whose members **disagree on the label**, which are guaranteed errors somewhere), near-duplicate pairs within 5% of the median nearest-neighbour distance, train-test **leakage** pairs (standardised with the training statistics), and text near-duplicates via 3-word shingles, MinHash, LSH banding and an exact-Jaccard verification pass (no false positives).
5. **Outliers.** Mean k-NN distance scored *within each labelled class* with a robust (median/MAD) z, so a point that is ordinary globally but far from its own class is found.
6. **Report.** Markdown or JSON: class balance (given vs estimated true share), flagged rows with the suggested label and probabilities, duplicate groups, outliers. Flagged rows are *candidates for a human*, not verdicts.

## Measured

Synthetic data with injected noise, so the truth is known. Two datasets (5-class overlapping Gaussians, a model trained on clean labels reaches about 0.89 test accuracy; 4 classes that are each two far-apart modes, which a linear model cannot separate), two probability models, five noise settings, 3 seeds, 1,800 training rows, 600 clean test rows. Full table in [`experiments/results.md`](experiments/results.md). A slice:

| data | model | noise | true -> est. rate | AUROC (CV margin) | AUROC in-sample loss | AUROC kNN-disagree | F1 (`both`) | test acc: noisy -> pruned (oracle clean) |
|---|---|---|---|---|---|---|---|---|
| blobs | softmax | symmetric 20% | 0.20 -> 0.24 | 0.984 | 0.986 | 0.969 | 0.861 | 0.879 -> 0.888 (0.895) |
| blobs | softmax | pairwise 30% | 0.30 -> 0.39 | 0.863 | 0.846 | 0.866 | 0.681 | 0.724 -> 0.824 (0.895) |
| blobs | kNN | pairwise 30% | 0.30 -> 0.37 | 0.918 | 0.846 | 0.866 | 0.764 | 0.724 -> 0.871 (0.895) |
| two-mode | softmax | symmetric 20% | 0.20 -> 0.33 | 0.933 | 0.937 | 0.977 | 0.714 | 0.772 -> 0.773 (0.777) |
| two-mode | kNN | symmetric 20% | 0.20 -> 0.23 | 0.983 | 0.937 | 0.977 | 0.858 | 0.772 -> 0.778 (0.777) |
| two-mode | kNN | pairwise 30% | 0.30 -> 0.35 | 0.934 | 0.799 | 0.880 | 0.787 | 0.675 -> 0.772 (0.777) |

What to take from it, including what did **not** work:

- **The model has to fit the data.** On the two-mode data the linear model's probabilities are poor and so is everything built on them (symmetric 20%: AUROC 0.933 with softmax, 0.983 with kNN; the noise rate is overestimated, 0.33 vs 0.20). Confident learning is model-agnostic, but it is only as good as the out-of-sample probabilities you feed it.
- **Structured noise is where it pays.** When labels are systematically confused (pairwise 30%), a model trained on the noisy labels drops to 0.72 (blobs) / 0.68 (two-mode) test accuracy and pruning the flagged rows recovers most of it (blobs 0.82 with softmax, 0.87 with kNN probabilities; oracle 0.895; two-mode 0.73 / 0.77, oracle 0.777). With random (symmetric) noise, a classifier is already fairly robust and pruning buys only 0.5-1.5 points.
- **Honest baselines are strong.** On the blob data a low-capacity in-sample softmax loss ranks errors as well as the cross-validated margin (0.986 vs 0.984): a regularised linear model cannot memorise noise, so cross-validation adds nothing there. In-sample loss falls to 0.80-0.85 under pairwise noise, which is the case cross-validation is for. The kNN-disagreement baseline even wins on the two-mode data against a softmax detector.
- **The noise-rate estimate overshoots** (0.20 -> 0.24, 0.30 -> 0.39) because genuinely ambiguous rows near class borders also look like confident-joint off-diagonals. Treat it as an upper-ish bound.
- **`both` is not always best**: it is the most conservative rule (highest precision); `prune_by_noise_rate` usually has the better F1 here. Single-seed gaps of a couple of hundredths are noise (3 seeds).
- **Everything is synthetic and the noise is injected at random**, so this says nothing about how real annotators fail (they are confused by hard examples, not uniformly). The audit is a way to prioritise a human review, not a replacement for one.

## Layout

```
src/datadoctor/  models.py cv.py confident.py dups.py outliers.py synth.py evaluate.py audit.py io.py __main__.py
tests/           41 tests: hand-computed confident joint, fold stratification, out-of-sample check,
                 metric edge cases (ties, AP by hand), noise injection exactness, duplicates/leakage/MinHash, CLI
experiments/     benchmark.py -> results.md
```

Tests: `PYTHONPATH=src python -m pytest -q` (about 2 s).

## HTML report

`python -m datadoctor audit data.csv --label y --html report.html` also writes a single self-contained HTML file (class balance bars, label-issue table, noise-matrix heatmap, duplicates, outliers). No JavaScript or external assets, so it can be attached to a PR or CI run as-is.

## Calibration (v0.2)

`datadoctor.calibrate(P, y)` fits one temperature on out-of-sample probabilities (golden-section search on NLL) and returns calibrated probabilities. It helps when the model is badly under- or over-confident and does nothing for a calibrated one; see `experiments/calibration_results.md` for the small, honestly mixed results.

### Instance-dependent noise (hard mode)

`inject_instance_noise` concentrates flips near class borders (each flipped row takes its nearest other class). Same 20% rate, 3 seeds, blobs ([`experiments/instance_noise_results.md`](experiments/instance_noise_results.md)):

| noise | model | true rate | est. rate | precision | recall | F1 |
|---|---|---|---|---|---|---|
| symmetric | softmax | 0.200 | 0.235 | 0.858 | 0.859 | 0.858 |
| symmetric | kNN 15 | 0.200 | 0.243 | 0.822 | 0.856 | 0.838 |
| instance | softmax | 0.200 | 0.057 | 0.960 | 0.226 | 0.363 |
| instance | kNN 15 | 0.200 | 0.072 | 0.871 | 0.258 | 0.396 |

Finding: with border-concentrated noise the detector stays precise (0.87-0.96) but recall collapses to ~0.23-0.26 and the noise rate is *under*-estimated (0.06-0.07 vs 0.20). A mislabelled border row is indistinguishable from a genuinely ambiguous one, so a model fitted on the noisy labels agrees with them. Treat the estimate as a lower bound under realistic human-style errors; uniform-noise benchmarks flatter every detector here.

## Diffing audits

```bash
datadoctor audit data.csv --label y --json --out before.json
# ...relabel flagged rows...
datadoctor audit data_fixed.csv --label y --json --out after.json
datadoctor diff before.json after.json --fail-on-regression
```

Reports the change in flagged share, CV accuracy and estimated noise rate, plus which label issues were resolved, introduced or persisted (rows matched by index, so keep row order stable). `--fail-on-regression` exits 1 if the flagged share grew, for CI.

## Regression targets

For a continuous target there is no confident joint, so `datadoctor.regression` uses
out-of-fold residuals scaled by a local robust spread (MAD inside prediction-quantile bins):

```python
from datadoctor.regression import find_target_issues, inject_target_noise
issues, scores, oof_pred = find_target_issues(X, y, threshold=3.5)
```

On a synthetic linear task with 8% of targets shifted by 2-4 std, the unit tests require
>80% precision and >70% recall and <3% flagged on clean data. The detector is only as good as
the model: a linear ridge cannot separate nonlinearity from corruption, so pass `model_factory`
(any object with `fit(X, y)` / `predict(X)`) for non-linear targets. Synthetic data only.


## Multi-label targets

`datadoctor.multilabel` runs confident learning per label (one-vs-rest) and returns an
n x K mask of cells whose tag looks missing (given 0) or spurious (given 1):

```python
from datadoctor import cross_val_probs_multilabel, find_multilabel_issues, rank_multilabel_issues
P = cross_val_probs_multilabel(X, Y)          # out-of-fold P(label k = 1)
mask = find_multilabel_issues(P, Y)
worst = rank_multilabel_issues(P, Y, mask, top=20)
```

Synthetic check (6 labels, 1500 examples, each label a noisy linear function of X, 3 seeds, random
cell flips): precision 0.85-0.91 and recall 0.83-0.91 at 2%, 5% and 10% noise. Caveat: the labels are
linear in X and the detector's model is linear, so this is the easy case; real tags correlated with
each other or non-linear in the features will score lower. Label co-occurrence is not yet used.

## Co-occurrence-aware multi-label scoring

`cooccurrence_refine(P, Y)` stacks a second-stage model over the logits of all first-stage
out-of-fold label probabilities (cross-validated, blended with the originals), so tags that
travel together inform each other. On `make_cooccurring_labels` (label 2 = AND of labels 0 and 1)
it lowers the log loss of the conjunction label (tested). **Honest result:** on issue detection
with 5% injected flips over 3 seeds, F1 moved 0.731->0.738, 0.834->0.834, 0.726->0.717 (base->stack),
i.e. noise. Neighbour columns carry the same noise, which cancels the gain. Treat it as a
better-calibrated probability, not a better detector, on this data.
