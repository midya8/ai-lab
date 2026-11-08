"""Command line: `python -m reviewkit review|rules|bench ...`."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional

from .bench import format_bench, load_cases, run_bench
from .github import build_review, post_review
from .llm import CommandProvider, OpenAICompatProvider, Provider
from .review import Reviewer, load_config, render_summary
from .rules import RULE_TABLE, SEVERITIES


def _read_diff(args) -> str:
    if args.diff:
        return sys.stdin.read() if args.diff == "-" else open(args.diff, encoding="utf-8").read()
    if args.base:
        cmd = ["git", "-C", args.root, "diff", "--no-color", "--unified=3", f"{args.base}...{args.head}"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"git diff failed: {proc.stderr.strip()}")
        return proc.stdout
    raise SystemExit("give --diff FILE (or -) or --base REF")


def _provider(args) -> Optional[Provider]:
    if args.provider == "none":
        return None
    if args.provider == "cmd":
        if not args.cmd:
            raise SystemExit("--provider cmd needs --cmd 'program args'")
        return CommandProvider(args.cmd.split())
    if not (args.base_url and args.model):
        raise SystemExit("--provider openai needs --base-url and --model")
    return OpenAICompatProvider(args.base_url, args.model, os.environ.get(args.api_key_env, ""))


def cmd_review(args) -> int:
    cfg = load_config(args.config)
    if args.max_comments:
        cfg.max_comments = args.max_comments
    if args.min_severity:
        cfg.min_severity = args.min_severity
    cfg.validate()
    result = Reviewer(cfg, _provider(args), root=args.root).review(_read_diff(args))

    if args.format == "json":
        print(json.dumps({"findings": [f.__dict__ | {"fp": f.fp} for f in result.all_findings],
                          "notes": result.notes, "stats": result.stats}, indent=2))
    elif args.format == "github":
        print(json.dumps(build_review(result), indent=2))
    elif args.format == "markdown":
        print(render_summary(result))
    else:
        for f in result.all_findings:
            print(f"{f.path}:{f.line}: [{f.severity}] {f.rule if f.source == 'rule' else 'model'}: {f.message}")
        for n in result.notes:
            print(f"note: {n}")
        print(f"{len(result.all_findings)} finding(s) in {result.files_reviewed} file(s)")

    if args.post:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise SystemExit("--post needs GITHUB_TOKEN in the environment")
        repo, _, pr = args.post.partition("#")
        out = post_review(result, repo, int(pr), token, commit_id=args.commit)
        print(f"posted: {out}", file=sys.stderr)
    if result.gate(args.fail_on):
        print(f"failing: findings at or above '{args.fail_on}'", file=sys.stderr)
        return 1
    return 0


def cmd_rules(args) -> int:
    for rid, (sev, msg) in sorted(RULE_TABLE.items(), key=lambda kv: (-SEVERITIES.index(kv[1][0]), kv[0])):
        print(f"{rid:26} {sev:9} {msg.splitlines()[0][:90]}")
    return 0


def cmd_bench(args) -> int:
    print(format_bench(run_bench(load_cases(args.cases))))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reviewkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="review a diff")
    r.add_argument("--diff", help="unified diff file, or - for stdin")
    r.add_argument("--base", help="git ref to diff against (uses base...head)")
    r.add_argument("--head", default="HEAD")
    r.add_argument("--root", default=".", help="checkout of the PR head (enables AST rules)")
    r.add_argument("--config", default=".reviewkit.json")
    r.add_argument("--format", choices=["text", "markdown", "json", "github"], default="text")
    r.add_argument("--fail-on", choices=SEVERITIES, help="exit 1 if any finding is at/above this severity")
    r.add_argument("--max-comments", type=int)
    r.add_argument("--min-severity", choices=SEVERITIES)
    r.add_argument("--provider", choices=["none", "openai", "cmd"], default="none")
    r.add_argument("--base-url")
    r.add_argument("--model")
    r.add_argument("--api-key-env", default="REVIEWKIT_API_KEY")
    r.add_argument("--cmd")
    r.add_argument("--post", metavar="OWNER/REPO#PR", help="post the review (needs GITHUB_TOKEN)")
    r.add_argument("--commit", help="head commit SHA for the review")
    r.set_defaults(fn=cmd_review)
    sub.add_parser("rules", help="list rules").set_defaults(fn=cmd_rules)
    b = sub.add_parser("bench", help="score the rules on a labelled case file")
    b.add_argument("cases")
    b.set_defaults(fn=cmd_bench)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
