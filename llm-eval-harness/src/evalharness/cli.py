import argparse
import json
import sys

from .backends import get_backend
from .report import compare, render_markdown
from .runner import load_suite, run_suite


def main(argv=None):
    p = argparse.ArgumentParser(prog="evalharness")
    p.add_argument("suite")
    p.add_argument("--backend", default="rules", choices=["echo", "rules", "http"])
    p.add_argument("--url"), p.add_argument("--model"), p.add_argument("--api-key", default="")
    p.add_argument("--save", help="write run JSON here")
    p.add_argument("--baseline", help="prior run JSON; exit 1 on regression")
    p.add_argument("--summary-file", help="append Markdown report here (e.g. $GITHUB_STEP_SUMMARY)")
    a = p.parse_args(argv)
    kw = {"url": a.url, "model": a.model, "api_key": a.api_key} if a.backend == "http" else {}
    run = run_suite(get_backend(a.backend, **kw), load_suite(a.suite))
    diff = None
    if a.baseline:
        with open(a.baseline) as f:
            diff = compare(json.load(f), run)
    if a.save:
        with open(a.save, "w") as f:
            json.dump(run, f, indent=2)
    md = render_markdown(run, diff)
    if a.summary_file:
        with open(a.summary_file, "a") as f:
            f.write(md + "\n")
    print(md)
    return 0 if diff is None or diff["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
