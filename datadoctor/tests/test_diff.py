import json

from datadoctor.__main__ import main
from datadoctor.audit import audit
from datadoctor.diff import diff_audits, render_diff
from datadoctor.synth import inject_label_noise, make_blobs


def _pair():
    X, y = make_blobs(n=300, seed=3)
    yn, flipped = inject_label_noise(y, 0.15, seed=3)
    a = audit(X, yn)
    fixed = yn.copy()
    fixed[flipped] = y[flipped]
    b = audit(X, fixed)
    return a, b


def test_cleaning_resolves_issues():
    a, b = _pair()
    d = diff_audits(a, b)
    assert d["same_rows"]
    assert len(d["resolved"]) > len(d["introduced"])
    assert d["flagged_rate_delta"] < 0 and d["cv_accuracy_delta"] > 0


def test_self_diff_is_empty():
    a, _ = _pair()
    d = diff_audits(a, a)
    assert not d["resolved"] and not d["introduced"] and d["flagged_rate_delta"] == 0
    assert len(d["persisting"]) == a["n_label_issues"] or len(d["persisting"]) == len(a["label_issues"])


def test_row_count_change_warns():
    a, b = _pair()
    b = dict(b, n=b["n"] + 10)
    assert "row count changed" in render_diff(diff_audits(a, b))


def test_cli_fail_on_regression(tmp_path):
    a, b = _pair()
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert main(["diff", str(pa), str(pb), "--fail-on-regression"]) == 0
    assert main(["diff", str(pb), str(pa), "--fail-on-regression"]) == 1
