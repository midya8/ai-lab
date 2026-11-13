import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logsentry.cli import main


def test_demo_incidents_text(capsys):
    assert main(["--demo", "--incidents"]) == 0
    out = capsys.readouterr().out
    assert "incidents" in out.splitlines()[0]
    assert any(l.startswith("[") for l in out.splitlines()[1:])


def test_demo_incidents_json_fewer_than_anomalies(capsys):
    main(["--demo"]); n_anoms = int(capsys.readouterr().out.split()[2])
    main(["--demo", "--incidents", "--json"])
    inc = json.loads(capsys.readouterr().out)
    assert 1 <= len(inc) <= n_anoms
    assert sum(i["anomalies"] for i in inc) == n_anoms
    assert all(i["severity"] in ("low", "medium", "high") for i in inc)


def test_gap_zero_splits_more(capsys):
    main(["--demo", "--incidents", "--json", "--gap", "100"]); wide = len(json.loads(capsys.readouterr().out))
    main(["--demo", "--incidents", "--json", "--gap", "0"]); tight = len(json.loads(capsys.readouterr().out))
    assert wide <= tight
