import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logsentry import parse_line, mine_template, detect, generate
from logsentry.cli import main


def test_template_masks_variables():
    a = mine_template("GET /api/users/42 200 15ms from 10.0.1.7")
    b = mine_template("GET /api/users/977 200 301ms from 10.0.9.250")
    assert a == b and "<IP>" in a and "<NUM>" in a


def test_parse_line_and_garbage():
    e = parse_line("2025-01-01T12:00:00 ERROR disk 93% full")
    assert e.level == "ERROR" and e.template == "disk <NUM>% full"
    assert parse_line("not a log") is None


def events(**kw):
    return [e for e in map(parse_line, generate(**kw)) if e]


def test_clean_logs_have_no_spike_or_burst():
    kinds = {a.kind for a in detect(events(incidents=False))}
    assert not kinds & {"volume_spike", "error_burst", "volume_drop"}


def test_incidents_detected():
    found = detect(events())
    kinds = {a.kind for a in found}
    assert {"volume_spike", "error_burst", "new_template"} <= kinds
    spike = next(a for a in found if a.kind == "volume_spike")
    assert spike.window.minute == 20
    burst = next(a for a in found if a.kind == "error_burst")
    assert burst.window.minute == 25


def test_empty_and_cli(capsys):
    assert detect([]) == []
    assert main(["--demo"]) == 0
    assert "anomalies" in capsys.readouterr().out
