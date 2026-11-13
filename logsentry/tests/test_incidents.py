import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datetime import datetime, timedelta
from logsentry import parse_line, detect, generate, Anomaly
from logsentry.incidents import group_incidents

T0 = datetime(2024, 1, 1, 12, 0, 0)


def A(i, kind="volume_spike", score=5.0):
    return Anomaly(kind, T0 + timedelta(seconds=60 * i), score, "")


def test_empty():
    assert group_incidents([]) == []


def test_chain_merges_and_gap_splits():
    inc = group_incidents([A(0), A(2), A(4), A(10)])
    assert len(inc) == 2
    assert (inc[0].start, inc[0].end) == (T0, T0 + timedelta(minutes=4))
    assert len(inc[1].anomalies) == 1


def test_unsorted_input_and_counts():
    inc = group_incidents([A(1, "error_burst", 0.5), A(0, "volume_spike", 9.0), A(0, "new_template", 1.0)])
    assert len(inc) == 1
    assert inc[0].kinds == {"volume_spike": 1, "error_burst": 1, "new_template": 1}
    assert inc[0].peak == 9.0 and inc[0].severity == "high"


def test_severity_levels():
    assert group_incidents([A(0, score=4.0)])[0].severity == "low"
    assert group_incidents([A(0, score=9.0)])[0].severity == "medium"
    assert group_incidents([A(0, score=-25.0)])[0].severity == "high"


def test_summary_text():
    s = group_incidents([A(0, score=9.0)])[0].summary()
    assert "medium" in s and "volume_spike x1" in s


def test_synthetic_run_collapses_alerts():
    events = [e for e in map(parse_line, generate(minutes=30, seed=0)) if e]
    an = detect(events)
    inc = group_incidents(an)
    assert len(inc) <= len(an)
    assert sum(len(i.anomalies) for i in inc) == len(an)
