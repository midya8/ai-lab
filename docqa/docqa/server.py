"""Stdlib HTTP service for DocQA.

GET /health -> {"status": "ok", "chunks": N}
POST /ask  {"question": "..."} -> QAResult dict (200 even on abstention)
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .qa import DocQA

MAX_BODY = 16 * 1024


def make_handler(qa: DocQA):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "chunks": len(qa.retriever.chunks)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "bad Content-Length"})
            if n > MAX_BODY:
                return self._send(413, {"error": "body too large"})
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            q = data.get("question") if isinstance(data, dict) else None
            if not isinstance(q, str) or not q.strip():
                return self._send(400, {"error": "question must be a non-empty string"})
            self._send(200, qa.ask(q).to_dict())

        def log_message(self, *a):
            pass

    return Handler


def make_server(qa: DocQA, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(qa))
