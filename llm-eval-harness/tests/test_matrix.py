import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evalharness import EchoBackend, RuleBasedBackend, load_suite, run_matrix, matrix_table, render_matrix_markdown
from evalharness.cli import main

SUITE = os.path.join(os.path.dirname(__file__), "..", "suites", "basic.json")


def test_matrix_ranks_rules_over_echo():
    m = run_matrix([EchoBackend(), RuleBasedBackend()], load_suite(SUITE))
    t = matrix_table(m)
    assert t["backends"] == ["echo", "rules"]
    assert t["winners"] == ["rules"] and t["means"][1] > t["means"][0]
    assert len(t["rows"]) == len(load_suite(SUITE)["cases"])
    assert t["contested"]


def test_tie_lists_both_winners():
    t = matrix_table({"suite": "s", "runs": [
        {"backend": "a", "mean_score": 1.0, "results": [{"id": "x", "score": 1.0}]},
        {"backend": "b", "mean_score": 1.0, "results": [{"id": "x", "score": 1.0}]}]})
    assert t["winners"] == ["a", "b"] and t["contested"] == []


def test_duplicate_backend_names_rejected():
    with pytest.raises(ValueError):
        run_matrix([EchoBackend(), EchoBackend()], load_suite(SUITE))


def test_markdown_and_cli(capsys, tmp_path):
    f = tmp_path / "s.md"
    assert main([SUITE, "--matrix", "echo,rules", "--summary-file", str(f)]) == 0
    out = capsys.readouterr().out
    assert "| case | echo | rules |" in out and "Best: rules" in out
    assert f.read_text().startswith("# Matrix")


def test_tag_table_means_and_best():
    from evalharness import tag_table
    m = {"suite": "s", "runs": [
        {"backend": "a", "results": [{"id": "1", "score": 1.0, "tags": ["x"]}, {"id": "2", "score": 0.0, "tags": ["x", "y"]}]},
        {"backend": "b", "results": [{"id": "1", "score": 0.0, "tags": ["x"]}, {"id": "2", "score": 1.0, "tags": ["x", "y"]}]}]}
    rows = {r["tag"]: r for r in tag_table(m)["rows"]}
    assert rows["x"]["scores"] == [0.5, 0.5] and rows["x"]["best"] == ["a", "b"]
    assert rows["y"]["scores"] == [0.0, 1.0] and rows["y"]["best"] == ["b"]


def test_markdown_has_tag_section(capsys):
    main([SUITE, "--matrix", "echo,rules"])
    assert "## By tag" in capsys.readouterr().out


def test_parse_http_spec_and_errors():
    from evalharness.backends import parse_backend_spec, HTTPBackend
    b = parse_backend_spec("http@http://localhost:11434/v1/chat/completions#llama3", "k")
    assert isinstance(b, HTTPBackend) and b.model == "llama3" and b.api_key == "k"
    assert b.url.endswith("/chat/completions") and b.name == "http:llama3"
    assert parse_backend_spec(" rules ").name == "rules"
    for bad in ("http@nohash", "http@ftp://x#m", "http@http://x#", "nope"):
        with pytest.raises(ValueError):
            parse_backend_spec(bad)


def test_matrix_cli_with_live_http_backend(capsys):
    import json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps({"choices": [{"message": {"content": "Paris"}}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out)))
            self.end_headers(); self.wfile.write(out)
        def log_message(self, *a): pass

    os.environ["no_proxy"] = "127.0.0.1"
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
        assert main([SUITE, "--matrix", f"rules,http@{url}#stub"]) == 0
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "rules" in out and "http:stub" in out
