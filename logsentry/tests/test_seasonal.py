import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from logsentry import parse_line, detect
from logsentry.seasonal import detect_seasonal, generate_seasonal, window_counts


def evs(**kw):
    return [e for e in map(parse_line, generate_seasonal(**kw)) if e]


def test_plain_detector_false_alarms_on_clean_cycle():
    flagged = [a for a in detect(evs()) if a.kind in ("volume_spike", "volume_drop")]
    assert len(flagged) > 0


def test_seasonal_quiet_on_clean_cycle():
    assert detect_seasonal(evs()) == []


def test_seasonal_finds_incident_at_peak_phase():
    idx = 3 * 24 + 6  # near the sinusoid peak, cycle 4
    found = detect_seasonal(evs(incident=(idx, 3)))
    assert [a.kind for a in found] == ["seasonal_spike"]
    assert found[0].window.minute == 18  # window index 78 = 01:18


def test_seasonal_finds_drop():
    found = detect_seasonal(evs(incident=(4 * 24 + 6, 0.05)))
    assert any(a.kind == "seasonal_drop" for a in found)


def test_needs_history_and_validates():
    assert detect_seasonal(evs(cycles=2)) == []  # only 1 prior cycle at most
    with pytest.raises(ValueError):
        detect_seasonal([], period=1)


def test_window_counts_dense():
    e = evs(cycles=1)
    assert len(window_counts(e)) == 24
