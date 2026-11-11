import io
import json

from codesearch.__main__ import main
from codesearch.server import Server, serve


def _run(repo, lines):
    out = io.StringIO()
    serve(repo, None, io.StringIO("\n".join(lines) + "\n"), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_search_ping_stats_and_id_echo(repo):
    r = _run(repo, [json.dumps({"id": "a", "op": "ping"}),
                    json.dumps({"id": 2, "op": "search", "query": "retry with exponential backoff", "k": 3}),
                    json.dumps({"id": 3, "op": "stats"})])
    assert r[0] == {"id": "a", "ok": True, "result": "pong"}
    assert r[1]["id"] == 2 and r[1]["ok"]
    assert "retry_with_backoff" in json.dumps(r[1]["result"])
    assert r[2]["result"]["chunks"] > 0


def test_bad_requests_do_not_kill_server(repo):
    r = _run(repo, ["not json", "[1]", json.dumps({"id": 1, "op": "nope"}),
                    json.dumps({"id": 2, "op": "search"}),
                    json.dumps({"id": 3, "op": "search", "query": "x", "mode": "zzz"}),
                    json.dumps({"id": 4, "op": "similar", "ident": "does_not_exist_anywhere"}),
                    json.dumps({"id": 5, "op": "ping"})])
    assert [x["ok"] for x in r] == [False, False, False, False, False, False, True]
    assert r[-1]["id"] == 5


def test_refresh_picks_up_new_file(repo):
    srv = Server(repo, None)
    q = {"op": "search", "query": "zebra_quantum_flux"}
    (repo / "zz.py").write_text("def zebra_quantum_flux():\n    '''Flux of zebras.'''\n    return 1\n")
    before = json.loads(srv.respond(json.dumps(q)))["result"]
    assert not any("zebra_quantum_flux" in h["qualname"] for h in before)
    json.loads(srv.respond(json.dumps({"op": "refresh"})))
    after = json.loads(srv.respond(json.dumps(q)))["result"]
    assert after[0]["qualname"] == "zebra_quantum_flux"


def test_cli_serve(repo, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"id": 1, "op": "ping"}) + "\n"))
    assert main(["serve", "--root", str(repo), "--no-store"]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "pong"
