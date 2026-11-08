# ticket-triage
Support-ticket triage classifier built from scratch in NumPy: unigram+bigram TF-IDF, softmax regression trained by gradient descent, confidence-based human escalation, confusion matrix and macro-F1.

```
pip install -r requirements.txt
PYTHONPATH=src python -m triage.cli "I was charged twice for my plan"
python -m pytest
```
Data is a deterministic synthetic generator (`triage.data`), so results are reproducible. k-fold CV, ECE and temperature-scaling calibration (`--cv 5`) in `triage.calibration`. Urgency head (`triage.urgency`): class-weighted logistic regression, recall-targeted threshold tuning, and `Triage.route()` combining category + urgency into a queue like `bug:urgent`.

Active learning (`triage.active`, `--active`): uncertainty sampling (least-confidence, margin, entropy) vs random picks the tickets a human should label next; `review_queue()` returns the ranked queue and `compare()` replays labeling budgets. `--active` output (5 seeds, accuracy at 8 -> 32 labels): random 0.507 -> 0.834 vs uncertainty strategies 0.904-0.927. Synthetic data, so treat as illustrative.
