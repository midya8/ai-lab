import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from triage.data import generate
from triage.model import TicketClassifier
from triage.conformal import conformal_quantile, ConformalTriage, evaluate


def noisy(recs, rate, seed):
    rng = np.random.default_rng(seed)
    cats = sorted({r["category"] for r in recs})
    return [rng.choice(cats) if rng.random() < rate else r["category"] for r in recs]


def test_quantile_rule():
    assert conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.2) == 8  # ceil(10*0.8)=8
    assert conformal_quantile([1, 2, 3], 0.1) == float("inf")


def test_coverage_with_label_noise():
    # 30% resampled labels make the task genuinely uncertain; coverage must still hold.
    covs = []
    for seed in range(8):
        recs = generate(1500, seed)
        y = noisy(recs, 0.3, seed + 100)
        t = [r["text"] for r in recs]
        clf = TicketClassifier(epochs=150).fit(t[:300], y[:300])
        ct = ConformalTriage(clf, 0.1).calibrate(t[300:800], y[300:800])
        covs.append(evaluate(ct.predict_sets(t[800:]), y[800:])["coverage"])
    assert abs(np.mean(covs) - 0.9) < 0.03 and min(covs) > 0.84


def test_route_actions_and_uncalibrated():
    recs = generate(400, 1)
    t = [r["text"] for r in recs]; y = [r["category"] for r in recs]
    clf = TicketClassifier(epochs=100).fit(t[:200], y[:200])
    ct = ConformalTriage(clf, 0.1)
    try:
        ct.predict_sets(t[:1]); assert False
    except RuntimeError:
        pass
    ct.calibrate(t[200:350], y[200:350])
    r = ct.route("I was charged twice for my plan")
    assert r["action"] in ("auto", "review", "escalate") and (r["action"] == "escalate") == (not r["candidates"])
