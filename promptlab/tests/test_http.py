import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from promptlab.models import HttpModel, make_judge, make_model
from promptlab.pairwise import make_pair_judge

os.environ["no_proxy"] = "127.0.0.1"


def serve(handler_fn):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(n))))
            code, headers, body = handler_fn(len(seen))
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen, f"http://127.0.0.1:{srv.server_port}/v1"


OK = {"choices": [{"message": {"content": "billing"}}]}


def test_success_request_shape_and_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    srv, seen, url = serve(lambda n: (200, {}, OK))
    try:
        assert HttpModel(url, "m1")("hello") == "billing"
    finally:
        srv.shutdown()
    path, headers, body = seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test-not-real"
    assert body["model"] == "m1" and body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_retries_429_then_succeeds():
    srv, seen, url = serve(lambda n: (429, {"Retry-After": "0"}, {}) if n < 3 else (200, {}, OK))
    try:
        assert HttpModel(url, "m", backoff=0.0)("x") == "billing"
    finally:
        srv.shutdown()
    assert len(seen) == 3


def test_400_fails_fast_without_retry():
    srv, seen, url = serve(lambda n: (400, {}, {"error": "bad"}))
    try:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            HttpModel(url, "m", backoff=0.0)("x")
    finally:
        srv.shutdown()
    assert len(seen) == 1


def test_gives_up_after_retries():
    srv, seen, url = serve(lambda n: (503, {}, {}))
    try:
        with pytest.raises(RuntimeError, match="HTTP 503 after 3 attempts"):
            HttpModel(url, "m", retries=2, backoff=0.0)("x")
    finally:
        srv.shutdown()
    assert len(seen) == 3


def test_bad_shape_is_error_and_key_never_leaks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    srv, _, url = serve(lambda n: (200, {}, {"nope": 1}))
    try:
        with pytest.raises(RuntimeError) as e:
            HttpModel(url, "m")("x")
    finally:
        srv.shutdown()
    assert "sk-secret-value" not in str(e.value)


def test_connection_refused_is_runtime_error():
    with pytest.raises(RuntimeError, match="connection error"):
        HttpModel("http://127.0.0.1:1/v1", "m", retries=0)("x")


def test_specs_parse():
    assert isinstance(make_model("http://127.0.0.1:9/v1#gpt"), HttpModel)
    assert make_model("http://h/v1#gpt").model == "gpt"
    assert isinstance(make_judge("https://h/v1#j"), HttpModel)
    assert isinstance(make_pair_judge("http://h/v1#j"), HttpModel)
    with pytest.raises(ValueError):
        make_model("http://h/v1")  # no model name


def test_cooldown_is_shared_and_extends_only():
    from promptlab.models import Cooldown
    now = [0.0]
    slept = []
    c = Cooldown(clock=lambda: now[0], sleep=slept.append)
    c.wait()
    assert slept == []
    c.penalize(5.0)
    c.penalize(2.0)  # a shorter penalty must not shorten the deadline
    c.wait()
    assert slept == [5.0] and c.waits == 1
    now[0] = 6.0
    c.wait()
    assert slept == [5.0]


def test_429_on_one_thread_pauses_the_others():
    import time

    def handler(n):
        if n == 1:
            return 429, {"Retry-After": "1"}, {}
        return 200, {}, OK

    srv, seen, url = serve(handler)
    try:
        m1, m2 = HttpModel(url, "m"), HttpModel(url, "m")
        assert m1.cooldown is m2.cooldown
        out = []
        t0 = time.monotonic()
        a = threading.Thread(target=lambda: out.append(m1("x")))
        a.start()
        time.sleep(0.3)  # let the 429 land
        b = threading.Thread(target=lambda: out.append(m2("y")))
        b.start()
        a.join(); b.join()
        elapsed = time.monotonic() - t0
    finally:
        srv.shutdown()
    assert out == ["billing", "billing"]
    # the second thread's first request waited for the shared deadline, so the server saw
    # only one early request before the cooldown, then both retried after it
    assert elapsed >= 0.9
    assert len(seen) == 3
    assert m1.cooldown.waits >= 1
