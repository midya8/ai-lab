import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from triage.data import generate
from triage.model import TicketClassifier
from triage.drift import psi, oov_rate, severity, drift_report


def fit():
    d = generate(600, seed=1)
    return TicketClassifier(epochs=100).fit([x["text"] for x in d], [x["category"] for x in d])


def test_psi_identical_zero_and_symmetric_positive():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0, abs=1e-9)
    a, b = psi([50, 30, 20], [10, 30, 60]), psi([10, 30, 60], [50, 30, 20])
    assert a > 0.25 and a == pytest.approx(b)


def test_severity_bands():
    assert [severity(v) for v in (0.05, 0.15, 0.4)] == ["stable", "moderate", "major"]


def test_same_distribution_no_alert():
    clf = fit()
    ref = [x["text"] for x in generate(300, seed=2)]
    new = [x["text"] for x in generate(300, seed=3)]
    r = drift_report(clf, ref, new)
    assert not r["alert"] and r["psi_severity"] == "stable"


def test_vocabulary_shift_detected():
    clf = fit()
    ref = [x["text"] for x in generate(200, seed=2)]
    new = ["kubernetes ingress controller returns gateway timeout on webhook retries"] * 50
    r = drift_report(clf, ref, new)
    assert r["oov_new"] > 0.5 and r["alert"]


def test_category_mix_shift_detected():
    clf = fit()
    d = generate(1200, seed=4)
    ref = [x["text"] for x in d[:300]]
    new = [x["text"] for x in d if x["category"] == "billing"][:200]
    assert drift_report(clf, ref, new)["psi"] > 0.25


def test_empty_batch_rejected():
    with pytest.raises(ValueError):
        drift_report(fit(), [], ["x"])
