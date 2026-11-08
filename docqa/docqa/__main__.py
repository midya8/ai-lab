import argparse
import json
import sys

from .evaluate import run_eval
from .qa import DocQA


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="docqa", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "eval"):
        s = sub.add_parser(name)
        s.add_argument("docs")
        s.add_argument("arg", help="question (ask) or labelled JSON file (eval)")
        s.add_argument("--min-coverage", type=float, default=0.6)
        s.add_argument("--json", action="store_true")
    sv = sub.add_parser("serve", help="run the HTTP service")
    sv.add_argument("docs")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--min-coverage", type=float, default=0.6)
    sv.add_argument("--cache-size", type=int, default=256, help="answer cache entries (0 disables)")
    sv.add_argument("--cache-ttl", type=float, default=300.0, help="cache TTL seconds (0 = no expiry)")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .cache import AnswerCache
        from .server import make_server
        try:
            cache = (AnswerCache(a.cache_size, a.cache_ttl or None)
                     if a.cache_size > 0 else None)
            srv = make_server(DocQA.from_dir(a.docs, min_coverage=a.min_coverage),
                              a.host, a.port, cache=cache)
        except (ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"docqa serving on http://{a.host}:{a.port}  (POST /ask, GET /stats, /metrics)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    try:
        qa = DocQA.from_dir(a.docs, min_coverage=a.min_coverage)
        if a.cmd == "ask":
            r = qa.ask(a.arg)
            print(json.dumps(r.to_dict(), indent=2) if a.json else r.text)
            return 0 if r.answered else 1
        res = run_eval(qa, a.arg)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"answered correctly: {res['answered_correctly']}/{res['answerable']}")
        print(f"abstained correctly: {res['correctly_abstained']}/{res['unanswerable']}")
        for f in res["failures"]:
            print("FAIL:", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
