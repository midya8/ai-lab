import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logsentry.grafana import build_dashboard, main
from logsentry import metrics

def test_valid_json_and_unique_ids():
    d = json.loads(json.dumps(build_dashboard()))
    ids = [p["id"] for p in d["panels"]]
    assert len(ids) == len(set(ids)) == 5

def test_exprs_use_only_exported_metrics():
    src = open(metrics.__file__).read()
    exported = set(re.findall(r"# TYPE (\w+)", src))
    for p in build_dashboard()["panels"]:
        used = set(re.findall(r"logsentry_\w+", p["targets"][0]["expr"]))
        assert used and used <= exported

def test_writes_file(tmp_path):
    f = tmp_path / "d.json"
    main([str(f)])
    assert json.load(open(f))["uid"] == "logsentry"
