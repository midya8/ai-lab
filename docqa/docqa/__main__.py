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
    a = p.parse_args(argv)
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
