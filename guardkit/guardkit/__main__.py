"""guardkit CLI.

    python -m guardkit check [--policy P] [--audit PATH] [--json] [--dry-run]
                             [TEXT | --file F | --stdin]
    python -m guardkit eval cases.jsonl [--policy P] [--min-recall R]
                                        [--max-fpr F] [--json OUT] [--md OUT]
    python -m guardkit validate-policy policy.json

Exit codes: 0 = allowed / gates passed / policy valid; 1 = blocked /
gates failed; 2 = usage or policy error. ``check`` reads stdin when no
text or --file is given.
"""

from __future__ import annotations

import argparse
import json
import sys

from .audit import AuditLog
from .eval import evaluate_cases, load_cases
from .guard import Guard
from .policy import PolicyError, default_policy, load_policy

__all__ = ["main"]


def _policy_or_default(path: str | None):
    return load_policy(path) if path else default_policy()


def _cmd_check(args) -> int:
    try:
        policy = _policy_or_default(args.policy)
    except PolicyError as exc:
        print(f"guardkit: {exc}", file=sys.stderr)
        return 2
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print(f"guardkit: cannot read {args.file}: {exc}", file=sys.stderr)
            return 2
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()
    audit = AuditLog(args.audit) if args.audit else None
    guard = Guard(policy, audit=audit)
    decision = guard.check_input(text, dry_run=args.dry_run)
    if args.json:
        print(
            json.dumps(
                {
                    "action": decision.action,
                    "safe_text": decision.safe_text,
                    "score": round(decision.score, 3),
                    "reasons": decision.reasons,
                    "findings": decision.public_findings(),
                },
                ensure_ascii=False,
            )
        )
    else:
        print(f"action: {decision.action}")
        print(f"injection score: {decision.score:.2f}")
        for reason in decision.reasons:
            print(f"reason: {reason}")
        for finding in decision.public_findings():
            print(
                f"finding: {finding['rule_id']} "
                f"({finding['category']}/{finding['severity']}): "
                f"{finding['message']}"
            )
        if decision.action != "block":
            print("--- safe text ---")
            print(decision.safe_text)
    return 1 if decision.blocked else 0


def _cmd_eval(args) -> int:
    try:
        policy = _policy_or_default(args.policy)
        cases = load_cases(args.cases)
    except (PolicyError, ValueError, OSError) as exc:
        print(f"guardkit: {exc}", file=sys.stderr)
        return 2
    report = evaluate_cases(cases, policy)
    data = report.to_dict()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(report.to_markdown())
    print(f"cases: {report.total}")
    print(f"attack recall: {report.attack_recall:.3f} (min {args.min_recall:.3f})")
    print(f"pii caught rate: {report.pii_caught_rate:.3f}")
    print(f"false-positive rate: {report.false_positive_rate:.3f} (max {args.max_fpr:.3f})")
    for label, row in report.per_label().items():
        print(f"  {label}: {row['caught']}/{row['n']} caught ({row['rate']:.3f})")
    failed = (
        report.attack_recall < args.min_recall
        or report.false_positive_rate > args.max_fpr
    )
    if failed:
        print("guardkit: eval gates failed", file=sys.stderr)
    return 1 if failed else 0


def _cmd_validate_policy(args) -> int:
    try:
        load_policy(args.policy)
    except PolicyError as exc:
        print(f"guardkit: invalid policy: {exc}", file=sys.stderr)
        return 2
    print(f"policy valid: {args.policy}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guardkit", description="Runtime guardrails for LLM pipelines."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="guard one prompt")
    check.add_argument("text", nargs="?", help="text to guard (default: stdin)")
    check.add_argument("--file", help="read text from a file")
    check.add_argument("--policy", help="policy JSON file")
    check.add_argument("--audit", help="append decisions to this JSONL audit log")
    check.add_argument("--json", action="store_true", help="machine-readable output")
    check.add_argument(
        "--dry-run", action="store_true", help="decide without writing the audit log"
    )
    check.set_defaults(func=_cmd_check)

    ev = sub.add_parser("eval", help="run the guard over labelled cases")
    ev.add_argument("cases", help="JSONL case file")
    ev.add_argument("--policy", help="policy JSON file")
    ev.add_argument("--min-recall", type=float, default=0.9)
    ev.add_argument("--max-fpr", type=float, default=0.1)
    ev.add_argument("--json", help="write JSON report to this file")
    ev.add_argument("--md", help="write Markdown report to this file")
    ev.set_defaults(func=_cmd_eval)

    val = sub.add_parser("validate-policy", help="validate a policy JSON file")
    val.add_argument("policy", help="policy JSON file")
    val.set_defaults(func=_cmd_validate_policy)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
