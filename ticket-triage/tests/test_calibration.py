import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from triage.data import generate
from triage.calibration import (kfold_indices, cross_validate, expected_calibration_error,
                                fit_temperature, nll, CalibratedClassifier)


def test_kfold_partitions():
    seen = []
    for tr, te in kfold_indices(103, 5):
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == 103
        seen += list(te)
    assert sorted(seen) == list(range(103))


def test_cross_validate_high_accuracy():
    d = generate(400)
    r = cross_validate([x["text"] for x in d], [x["category"] for x in d], k=4)
    assert len(r["acc"]) == 4 and r["acc_mean"] > 0.9


def test_ece_perfect_and_overconfident():
    p = np.array([[1.0, 0], [0, 1.0]])
    assert expected_calibration_error(p, [0, 1]) == 0.0
    p = np.array([[0.99, 0.01]] * 4)
    assert expected_calibration_error(p, [0, 1, 1, 1]) > 0.7


def test_temperature_softens_overconfident_logits():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, 500)
    logits = rng.randn(500, 3) * 3
    logits[np.arange(500), y] += 3 * (rng.rand(500) < 0.6)  # only 60% informative
    logits *= 4  # overconfident
    T = fit_temperature(logits, y)
    assert T > 1.5
    assert nll(logits, y, T) < nll(logits, y, 1.0)


def test_calibrated_classifier_predicts():
    d = generate(500)
    c = CalibratedClassifier().fit([x["text"] for x in d[:400]], [x["category"] for x in d[:400]])
    pred = c.predict([x["text"] for x in d[400:]])
    acc = np.mean([p == x["category"] for p, x in zip(pred, d[400:])])
    assert acc > 0.9 and c.T > 0
    assert np.allclose(c.predict_proba(["refund me"]).sum(), 1)
