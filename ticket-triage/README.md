# ticket-triage
Support-ticket triage classifier built from scratch in NumPy: unigram+bigram TF-IDF, softmax regression trained by gradient descent, confidence-based human escalation, confusion matrix and macro-F1.

```
pip install -r requirements.txt
PYTHONPATH=src python -m triage.cli "I was charged twice for my plan"
python -m pytest
```
Data is a deterministic synthetic generator (`triage.data`), so results are reproducible. Next: urgency head, k-fold CV, calibration.
