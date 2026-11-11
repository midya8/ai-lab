"""JSON-lines server: one JSON request per input line, one JSON response per output line.

Meant for editor plugins and agents that want a long-lived process (index loaded once).
Requests: {"id": any, "op": "search"|"similar"|"stats"|"refresh"|"ping", ...params}
Responses: {"id": same, "ok": true, "result": ...} or {"id": same, "ok": false, "error": "..."}.
A malformed line never kills the server.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from .index import Index
from .search import MODES, Searcher
from .store import sync


class Server:
    def __init__(self, root: Path, store: Path | None, include=None):
        self.root, self.store, self.include = Path(root), store, include
        self.refresh()

    def refresh(self) -> dict:
        chunks, report = sync(self.root, self.store, include=self.include)
        self.index = Index.build(chunks)
        self.searcher = Searcher(self.index)
        return {"chunks": len(chunks), "sync": report.summary()}

    def handle(self, req: dict) -> object:
        op = req.get("op")
        if op == "ping":
            return "pong"
        if op == "refresh":
            return self.refresh()
        if op == "stats":
            kinds: dict[str, int] = {}
            for c in self.index.chunks:
                kinds[c.kind] = kinds.get(c.kind, 0) + 1
            return {"chunks": len(self.index.chunks), "kinds": kinds,
                    "files": len({c.path for c in self.index.chunks})}
        if op == "search":
            q = req.get("query")
            if not isinstance(q, str) or not q.strip():
                raise ValueError("search needs a non-empty string 'query'")
            mode = req.get("mode", "bm25f")
            if mode not in MODES:
                raise ValueError(f"unknown mode {mode!r}; choose from {list(MODES)}")
            hits = self.searcher.search(q, k=int(req.get("k", 8)), mode=mode,
                                        kinds=req.get("kinds"), path_prefix=req.get("path"),
                                        include_tests=bool(req.get("tests", False)))
            return [h.to_dict() for h in hits]
        if op == "similar":
            ident = req.get("ident")
            if not isinstance(ident, str):
                raise ValueError("similar needs a string 'ident'")
            try:
                hits = self.searcher.similar(ident, k=int(req.get("k", 8)))
            except KeyError as e:
                raise ValueError(e.args[0]) from None
            return [h.to_dict() for h in hits]
        raise ValueError(f"unknown op {op!r}")

    def respond(self, line: str) -> str:
        rid = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
            rid = req.get("id")
            return json.dumps({"id": rid, "ok": True, "result": self.handle(req)})
        except Exception as e:  # noqa: BLE001 - the server must survive any bad request
            return json.dumps({"id": rid, "ok": False, "error": str(e) or type(e).__name__})


def serve(root: Path, store: Path | None, stdin: IO[str], stdout: IO[str], include=None) -> int:
    srv = Server(root, store, include)
    n = 0
    for line in stdin:
        if not line.strip():
            continue
        stdout.write(srv.respond(line) + "\n")
        stdout.flush()
        n += 1
    return n
