import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datetime import datetime, timedelta
from logsentry.detect import Anomaly
from logsentry.alerts import WebhookNotifier

T = datetime(2025, 1, 1, 12, 0)

def A(kind="error_burst", m=0, score=2.0, d="x"):
    return Anomaly(kind, T + timedelta(minutes=m), score, d)

def test_sends_payload():
    got = []
    n = WebhookNotifier("u", sender=got.append)
    n.notify([A()])
    assert got[0]["kind"] == "error_burst" and "logsentry" in got[0]["text"]

def test_cooldown_dedupes_then_releases():
    got = []
    n = WebhookNotifier("u", cooldown=300, sender=got.append)
    n.notify([A(m=0), A(m=1), A(m=6)])
    assert len(got) == 2 and n.suppressed == 1

def test_different_detail_not_deduped():
    got = []
    n = WebhookNotifier("u", sender=got.append)
    n.notify([A(d="a"), A(d="b")])
    assert len(got) == 2

def test_filters_kind_and_score():
    got = []
    n = WebhookNotifier("u", kinds=["error_burst"], min_score=1.5, sender=got.append)
    n.notify([A("volume_spike"), A(score=1.0), A(score=2.0)])
    assert len(got) == 1

def test_failure_does_not_raise_and_retries():
    calls = []
    def bad(b):
        calls.append(b); raise OSError("down")
    n = WebhookNotifier("u", sender=bad)
    assert n.notify([A()]) == []
    n.notify([A(m=1)])
    assert len(calls) == 2 and n.sent == 0
