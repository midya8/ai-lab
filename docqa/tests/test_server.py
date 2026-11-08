import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

import pytest  # noqa: E402

from docqa import DocQA  # noqa: E402
from docqa.server import make_server  # noqa: E402


@pytest.fixture(scope="module")
def base():
    srv = make_server(DocQA.from_dir(DOCS), port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(url, data=None, raw=None):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    try:
        with OPENER.open(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health(base):
    code, d = call(base + "/health")
    assert code == 200 and d["status"] == "ok" and d["chunks"] > 0


def test_ask_answers_with_citation(base):
    code, d = call(base + "/ask", {"question": "How long do I have to return a bike?"})
    assert code == 200 and d["answered"] is True


def test_ask_abstains_with_200(base):
    code, d = call(base + "/ask", {"question": "What is the airspeed of a swallow?"})
    assert code == 200 and d["answered"] is False


def test_bad_requests(base):
    assert call(base + "/ask", {"question": "  "})[0] == 400
    assert call(base + "/ask", raw=b"{not json")[0] == 400
    assert call(base + "/nope")[0] == 404
