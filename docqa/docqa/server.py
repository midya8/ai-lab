"""Stdlib HTTP service for DocQA.

GET  /health  -> {"status": "ok", "chunks": N}
GET  /stats   -> request, outcome, cache and latency stats (JSON)
GET  /metrics -> Prometheus text exposition
POST /ask     {"question": "..."} -> QAResult dict (200 even on abstention);
              X-Cache: HIT|MISS header when a cache is enabled
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .cache import AnswerCache
from .metrics import Metrics
from .qa import DocQA

MAX_BODY = 16 * 1024
ROUTES = {"/ask", "/health", "/stats", "/metrics"}


def make_handler(qa: DocQA, cache: AnswerCache | None = None, metrics: Metrics | None = None):
    metrics = metrics if metrics is not None else Metrics()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload, *, ctype="application/json", headers=None,
                  route="other", t0=None, outcome=None, cache_result=None) -> None:
            body = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            metrics.observe(route if route in ROUTES else "other", code,
                            time.perf_counter() - (t0 or time.perf_counter()),
                            outcome=outcome, cache=cache_result)

        def do_GET(self):
            t0 = time.perf_counter()
            kw = {"route": self.path, "t0": t0}
            if self.path == "/health":
                self._send(200, {"status": "ok", "chunks": len(qa.retriever.chunks)}, **kw)
            elif self.path == "/stats":
                snap = metrics.snapshot()
                snap["cache"] = cache.stats() if cache is not None else None
                self._send(200, snap, **kw)
            elif self.path == "/metrics":
                self._send(200, metrics.prometheus(cache.stats() if cache is not None else None),
                           ctype="text/plain; version=0.0.4", **kw)
            else:
                self._send(404, {"error": "not found"}, **kw)

        def do_POST(self):
            t0 = time.perf_counter()
            kw = {"route": self.path, "t0": t0}
            if self.path != "/ask":
                return self._send(404, {"error": "not found"}, **kw)
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "bad Content-Length"}, **kw)
            if n > MAX_BODY:
                return self._send(413, {"error": "body too large"}, **kw)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"}, **kw)
            q = data.get("question") if isinstance(data, dict) else None
            if not isinstance(q, str) or not q.strip():
                return self._send(400, {"error": "question must be a non-empty string"}, **kw)
            cached = cache.get(q) if cache is not None else None
            if cached is not None:
                result = dict(cached, question=q)  # echo the caller's own phrasing
            else:
                result = qa.ask(q).to_dict()
                if cache is not None:
                    cache.put(q, result)
            headers = {"X-Cache": "HIT" if cached is not None else "MISS"} if cache is not None else None
            self._send(200, result, headers=headers,
                       outcome="answered" if result["answered"] else "abstained",
                       cache_result=("hit" if cached is not None else "miss") if cache is not None else None,
                       **kw)

        def log_message(self, *a):
            pass

    Handler.metrics = metrics
    Handler.cache = cache
    return Handler


def make_server(qa: DocQA, host: str = "127.0.0.1", port: int = 8000,
                cache: AnswerCache | None = None, metrics: Metrics | None = None
                ) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(qa, cache, metrics))
    srv.cache, srv.metrics = cache, srv.RequestHandlerClass.metrics
    return srv
