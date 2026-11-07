import os, sys, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logsentry import parse_line, generate
from logsentry.metrics import MetricsDetector, serve


def run():
    md = MetricsDetector()
    n = 0
    for e in filter(None, map(parse_line, generate())):
        md.feed(e); n += 1
    md.flush()
    return md, n


def test_counts_match():
    md, n = run()
    assert sum(md.events.values()) == n
    assert sum(md.anomalies.values()) > 0


def test_render_format():
    md, n = run()
    out = md.render()
    assert "# TYPE logsentry_events_total counter" in out
    assert 'logsentry_anomalies_total{kind="error_burst"}' in out
    assert f"logsentry_windows_closed_total {md.sd.n_closed}" in out
    assert out.endswith("\n")


def test_http_endpoint():
    md, _ = run()
    srv = serve(md, 0)
    try:
        port = srv.server_address[1]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        body = opener.open(f"http://127.0.0.1:{port}/metrics").read().decode()
        assert "logsentry_events_total" in body
    finally:
        srv.shutdown()
