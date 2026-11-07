"""evalkit command-line interface.

    python -m evalkit validate cases/demo.json
    python -m evalkit run cases/demo.json --adapter stub \\
        --fixtures fixtures/stub_responses.json --out runs/run.json --repeat 2
    python -m evalkit compare runs/baseline.json runs/run.json --min-score 0.85
    python -m evalkit report runs/run.json --md report.md --json report.json

Exit codes: 0 = all pass / no regression, 1 = failures or regression,
2 = invalid cases, bad adapter, or unusable run file.
"""

from __future__ import annotations

import argparse
import sys

from .adapters import AdapterError, build_adapter
from .cases import CaseError, filter_cases, load_cases
from .compare import compare_runs, format_compare
from .report import write_report
from .runner import RunResult, run_suite


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.golden_set)
    except CaseError as exc:
        print(f"invalid golden set: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {len(cases)} cases, {sum(c.check_count() for c in cases)} checks")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.golden_set)
    except CaseError as exc:
        print(f"invalid golden set: {exc}", file=sys.stderr)
        return 2
    cases = filter_cases(cases, only_tag=args.only, id_substring=args.filter)
    if not cases:
        print("no cases match --only/--filter", file=sys.stderr)
        return 2
    try:
        adapter = build_adapter(
            args.adapter,
            fixtures=args.fixtures,
            model=args.model,
            base_url=args.base_url,
        )
    except AdapterError as exc:
        print(f"adapter error: {exc}", file=sys.stderr)
        return 2
    try:
        result = run_suite(cases, adapter, repeat=args.repeat, seed=args.seed)
    except AdapterError as exc:
        print(f"adapter error: {exc}", file=sys.stderr)
        return 2
    if args.out:
        result.save(args.out)
        print(f"wrote {args.out}")
    print(
        f"overall score: {result.overall_score:.3f} "
        f"({result.passed_count} passed, {result.failed_count} failed, "
        f"{len(result.cases)} total)"
    )
    for case in sorted(result.cases, key=lambda c: (c.score, c.case_id)):
        if not case.passed:
            failed = "; ".join(c["name"] for c in case.checks if not c["passed"])
            print(f"  FAIL {case.case_id} ({case.score:.2f}): {failed}")
    if result.flaky_ids:
        print(f"  flaky: {', '.join(result.flaky_ids)}")
    return 0 if result.failed_count == 0 and not result.flaky_ids else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        baseline = RunResult.load(args.baseline)
        run = RunResult.load(args.run)
    except ValueError as exc:
        print(f"compare error: {exc}", file=sys.stderr)
        return 2
    result = compare_runs(baseline, run, min_score=args.min_score)
    result.baseline_path = args.baseline
    result.run_path = args.run
    print(format_compare(result))
    return 1 if result.regressed else 0


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        result = RunResult.load(args.run)
    except ValueError as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 2
    if not args.md and not args.json:
        print("nothing to write: pass --md and/or --json", file=sys.stderr)
        return 2
    for path in write_report(result, md_path=args.md, json_path=args.json):
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalkit", description="LLM evaluation harness for catching regressions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate a golden-set file")
    p.add_argument("golden_set")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("run", help="run a golden set against an adapter")
    p.add_argument("golden_set")
    p.add_argument("--adapter", choices=["stub", "openai"], default="stub")
    p.add_argument("--fixtures", help="stub fixtures file (required for stub)")
    p.add_argument("--model", help="model name for the openai adapter")
    p.add_argument("--base-url", help="base URL for the openai adapter")
    p.add_argument("--out", help="where to write the run JSON")
    p.add_argument("--only", help="run only cases with this tag")
    p.add_argument("--filter", help="run only cases whose id contains this string")
    p.add_argument("--repeat", type=int, default=1, help="repeats per case (flaky check)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("compare", help="diff two run files for regressions")
    p.add_argument("baseline")
    p.add_argument("run")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="fail if the run's overall score is below this")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("report", help="render Markdown/JSON reports from a run file")
    p.add_argument("run")
    p.add_argument("--md", help="Markdown report path")
    p.add_argument("--json", help="JSON report path")
    p.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
