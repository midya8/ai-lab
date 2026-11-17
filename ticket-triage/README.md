# ticket-triage

A support-ticket classifier in NumPy: unigram + bigram TF-IDF into softmax regression, with low-confidence tickets escalated to a human. It kept growing into a small triage system around that.

```bash
cd ticket-triage
pip install -r requirements.txt
PYTHONPATH=src python -m triage.cli "I was charged twice for my plan"
PYTHONPATH=src python -m triage.cli "..." --tags      # category + urgency + tags -> queue and SLA
PYTHONPATH=src python -m triage.sla_sim               # staffing simulation
python -m pytest -q
```

The data comes from a deterministic synthetic generator, so everything is reproducible but also easier than real tickets.

What's in it:

- Calibration with k-fold CV, ECE and temperature scaling.
- An urgency head with recall-targeted thresholds, and routing like `bug:urgent`.
- Active learning. Uncertainty sampling reaches 0.90-0.93 accuracy at 32 labels where random picks get 0.83.
- Multi-label tags (mobile, performance, security, enterprise) and rule-based priority and SLA routing.
- A discrete-event SLA simulation to find the fewest agents that meet a breach target.
- A label audit using confident learning from [`datadoctor`](../datadoctor). On 500 tickets with 10% flipped labels it finds them with 0.78 precision and 0.87 recall.
- Near-duplicate clustering, a drift monitor (PSI, out-of-vocabulary rate, confidence drop), and conformal prediction sets that decide between auto-route, review and escalate.

Everything is measured on synthetic data, so take the numbers as illustrative.
