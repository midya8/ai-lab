import json
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

import pytest  # noqa: E402

from docqa import DocQA  # noqa: E402
from docqa.cache import AnswerCache, normalize  # noqa: E402
from docqa.metrics import Metrics, percentile  # noqa: E402
from docqa.server import make_server  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_normalize_ignores_case_punct_space_but_not_order():
    assert normalize("  How LONG,  do I return?? ") == "how long do i return"
    assert normalize("return a bike") != normalize("a bike return")


def test_lru_eviction_and_recency():
    c = AnswerCache(max_size=2, ttl=None)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1          # a becomes most recent
    c.put("c", 3)                   # evicts b
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    assert c.stats()["evictions"] == 1


def test_ttl_expiry_with_fake_clock():
    clk = Clock()
    c = AnswerCache(max_size=4, ttl=10, clock=clk)
    c.put("q", "v")
    clk.t = 9.9
    assert c.get("q") == "v"
    clk.t = 10.0
    assert c.get("q") is None and len(c) == 0
    assert c.stats()["expirations"] == 1


def test_hit_rate_and_validation():
    c = AnswerCache()
    c.put("x", 1); c.get("x"); c.get("y")
    assert c.stats()["hit_rate"] == 0.5
    with pytest.raises(ValueError):
        AnswerCache(max_size=0)
    with pytest.raises(ValueError):
        AnswerCache(ttl=0)


def test_cache_thread_safety_smoke():
    c = AnswerCache(max_size=8, ttl=None)

    def work(i):
        for j in range(200):
            c.put(f"k{(i * j) % 20}", j)
            c.get(f"k{j % 20}")
    ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(c) <= 8


def test_percentile_nearest_rank():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == 5 and percentile(xs, 95) == 10 and percentile(xs, 0) == 1
    assert percentile([], 50) == 0.0
    with pytest.raises(ValueError):
        percentile(xs, 101)


def test_metrics_window_is_bounded_and_buckets_cumulative():
    m = Metrics(window=3)
    for s in (0.0005, 0.02, 0.3, 0.4, 2.0):
        m.observe("/ask", 200, s, outcome="answered", cache="miss")
    snap = m.snapshot()
    assert snap["latency_ms"]["count"] == 5
    assert snap["latency_ms"]["max"] == 2000.0   # window holds the last 3: 0.3, 0.4, 2.0
    text = m.prometheus()
    assert 'docqa_ask_seconds_bucket{le="0.001"} 1' in text
    assert 'docqa_ask_seconds_bucket{le="0.025"} 2' in text
    assert 'docqa_ask_seconds_bucket{le="+Inf"} 5' in text
    assert "docqa_ask_seconds_count 5" in text


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base, q):
    req = urllib.request.Request(base + "/ask", data=json.dumps({"question": q}).encode(), method="POST")
    with OPENER.open(req) as r:
        return r.headers.get("X-Cache"), json.loads(r.read())


def get(url):
    with OPENER.open(url) as r:
        return r.read().decode()


@pytest.fixture()
def served():
    cache = AnswerCache(max_size=16, ttl=None)
    srv = make_server(DocQA.from_dir(DOCS), port=0, cache=cache)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", cache
    srv.shutdown(); srv.server_close()


def test_server_cache_hit_echoes_callers_question(served):
    base, _ = served
    h1, r1 = post(base, "How long do I have to return a bike?")
    h2, r2 = post(base, "how long do i have to return a bike")
    assert (h1, h2) == ("MISS", "HIT")
    assert r2["question"] == "how long do i have to return a bike"
    assert {k: v for k, v in r1.items() if k != "question"} == {k: v for k, v in r2.items() if k != "question"}


def test_server_stats_and_prometheus(served):
    base, _ = served
    post(base, "How long do I have to return a bike?")
    post(base, "How long do I have to return a bike?")
    post(base, "What is the airspeed of a swallow?")
    s = json.loads(get(base + "/stats"))
    assert s["answered"] == 2 and s["abstained"] == 1
    assert s["cache_hits"] == 1 and s["cache_misses"] == 2
    assert s["cache"]["size"] == 2 and s["latency_ms"]["count"] == 3
    assert s["requests"]["/ask 200"] == 3
    prom = get(base + "/metrics")
    assert 'docqa_answers_total{outcome="abstained"} 1' in prom
    assert 'docqa_cache_requests_total{result="hit"} 1' in prom
    assert "docqa_cache_entries 2" in prom


def test_server_without_cache_has_no_xcache_header():
    srv = make_server(DocQA.from_dir(DOCS), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        h, r = post(base, "How long do I have to return a bike?")
        assert h is None and r["answered"]
        assert json.loads(get(base + "/stats"))["cache"] is None
    finally:
        srv.shutdown(); srv.server_close()


def test_unknown_routes_do_not_blow_up_label_cardinality(served):
    base, _ = served
    for p in ("/a", "/b", "/c"):
        try:
            get(base + p)
        except Exception:
            pass
    s = json.loads(get(base + "/stats"))
    assert s["requests"]["other 404"] == 3
