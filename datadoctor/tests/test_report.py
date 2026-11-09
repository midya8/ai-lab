import csv
import re

from datadoctor.__main__ import main
from datadoctor.audit import audit
from datadoctor.report import render_html
from datadoctor.synth import inject_label_noise, make_blobs


def _result():
    X, y = make_blobs(n=300, seed=1)
    yn, _ = inject_label_noise(y, 0.1, seed=1)
    return audit(X, yn, class_names=["a<b"] + [f"c{i}" for i in range(1, int(yn.max()) + 1)])


def test_html_is_well_formed_and_escaped():
    h = render_html(_result())
    assert h.startswith("<!doctype html>") and h.endswith("</html>")
    assert "a&lt;b" in h and "a<b" not in h
    assert h.count("<table") == h.count("</table>")
    assert "Estimated noise matrix" in h


def test_top_limits_rows():
    r = _result()
    h = render_html(r, top=2)
    if r["n_label_issues"] > 2:
        assert "more not shown" in h


def test_cli_writes_html(tmp_path):
    X, y = make_blobs(n=200, seed=2)
    p = tmp_path / "d.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "x1", "label"])
        for xi, yi in zip(X, y):
            w.writerow([xi[0], xi[1], int(yi)])
    out = tmp_path / "r.html"
    main(["audit", str(p), "--label", "label", "--html", str(out)])
    assert "<h1>Dataset audit</h1>" in out.read_text()
