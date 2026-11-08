from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .compare import check_regression, render_check_markdown, compare_runs, render_markdown
from .judge import calibrate_judge, load_calibration_items, render_calibration
from .models import make_judge, make_model
from .pairwise import (audit_judge, compare_outputs, load_pair_items, make_pair_judge, render_audit,
                       render_pairwise)
from .registry import Registry, RegistryError
from .runner import RunResult, load_cases, run_prompt


def _run(reg: Registry, args, ref: str) -> RunResult:
    v = reg.resolve(args.name, ref)
    return run_prompt(v, load_cases(args.cases), make_model(args.model),
                      samples=args.samples, threshold=args.threshold, judge=make_judge(args.judge))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promptlab", description="Prompt versioning, A/B testing and regression gates")
    ap.add_argument("--root", default=os.environ.get("PROMPTLAB_ROOT", "prompts"), help="registry directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_run_opts(p):
        p.add_argument("--cases", required=True, help="JSONL test cases")
        p.add_argument("--model", default="rule", help="rule | rule:<seed> | cmd:<command>")
        p.add_argument("--samples", type=int, default=1, help="model calls per case (non-deterministic models)")
        p.add_argument("--threshold", type=float, default=1.0, help="fraction of samples that must pass a case")
        p.add_argument("--judge", default=None, help="model for judge checks: rule | cmd:<command> (judge checks fail without one)")

    p = sub.add_parser("commit", help="store a new prompt version"); p.add_argument("name"); p.add_argument("--file", required=True); p.add_argument("-m", "--message", default="")
    sub.add_parser("list", help="list prompts")
    p = sub.add_parser("log", help="version history"); p.add_argument("name")
    p = sub.add_parser("show", help="print a version"); p.add_argument("name"); p.add_argument("--ref", default="latest")
    p = sub.add_parser("diff", help="unified diff of two versions"); p.add_argument("name"); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("tag", help="point a tag at a version"); p.add_argument("name"); p.add_argument("tag"); p.add_argument("--ref", default="latest")
    p = sub.add_parser("run", help="run a version over test cases"); p.add_argument("name"); p.add_argument("--ref", default="latest"); add_run_opts(p); p.add_argument("--out"); p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("compare", help="paired A/B comparison"); p.add_argument("name"); p.add_argument("a"); p.add_argument("b"); add_run_opts(p); p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--report", help="write Markdown report")
    p = sub.add_parser("baseline", help="save a run as the regression baseline"); p.add_argument("name"); p.add_argument("--ref", default="latest"); add_run_opts(p); p.add_argument("--out", required=True)
    p = sub.add_parser("check", help="gate a version against a baseline (exit 1 on failure)"); p.add_argument("name"); p.add_argument("--ref", default="latest"); add_run_opts(p)
    p.add_argument("--baseline", required=True); p.add_argument("--max-regressions", type=int, default=0); p.add_argument("--min-pass-rate", type=float); p.add_argument("--max-drop", type=float); p.add_argument("--summary", help="append a Markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)")

    p = sub.add_parser("calibrate", help="measure a judge's agreement with human scores")
    p.add_argument("--labels", required=True, help="JSONL of {id, rubric, output, human_score}")
    p.add_argument("--judge", default="rule", help="rule | cmd:<command>")
    p.add_argument("--min-score", type=int, default=4, help="score at or above which an output passes")
    p.add_argument("--votes", type=int, default=1, help="judge calls per item (lower median)")
    p.add_argument("--min-kappa", type=float, help="exit 1 if Cohen's kappa is below this (or undefined)")

    p = sub.add_parser("pairwise", help="pairwise LLM-judge A/B of two versions, each pair judged in both orders")
    p.add_argument("name"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--cases", required=True); p.add_argument("--model", default="rule")
    p.add_argument("--rubric", help="grading rubric (default: the first judge check's rubric in each case)")
    p.add_argument("--judge", default="rule", help="rule | rule-biased:<p>[:A|B] | cmd:<command>")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-inconsistent", type=float, help="exit 1 if more than this share of pairs flip when order is swapped")
    p.add_argument("--out", help="write per-pair verdicts as JSON")

    p = sub.add_parser("pair-audit", help="measure a pairwise judge against human preferences and check it for position bias")
    p.add_argument("--labels", required=True, help="JSONL of {id, rubric, a, b, human: a|b|tie}")
    p.add_argument("--judge", default="rule", help="rule | rule-biased:<p>[:A|B] | cmd:<command>")
    p.add_argument("--min-accuracy", type=float, help="exit 1 if reconciled accuracy is below this")
    p.add_argument("--max-inconsistent", type=float, help="exit 1 if the share of pairs answered differently in the two orders exceeds this")

    args = ap.parse_args(argv)
    reg = Registry(args.root)
    try:
        if args.cmd == "commit":
            v, created = reg.commit(args.name, Path(args.file).read_text(encoding="utf-8"), args.message)
            print(f"{'committed' if created else 'unchanged'}: {v.label}")
        elif args.cmd == "list":
            for n in reg.names():
                vs = reg.versions(n)
                print(f"{n}\tv{vs[-1].number}\t{len(vs)} version(s)")
        elif args.cmd == "log":
            tags = reg.tags_for(args.name)
            for v in reversed(reg.versions(args.name)):
                t = ",".join(k for k, n in sorted(tags.items()) if n == v.number)
                print(f"v{v.number}  {v.hash[:8]}  {v.created}  {v.message}" + (f"  [{t}]" if t else ""))
        elif args.cmd == "show":
            v = reg.resolve(args.name, args.ref)
            print(f"# {v.label}  vars: {', '.join(v.template.variables()) or '-'}\n{v.text}", end="")
        elif args.cmd == "diff":
            print(reg.diff(args.name, args.a, args.b) or "(no differences)")
        elif args.cmd == "tag":
            v = reg.tag(args.name, args.tag, args.ref)
            print(f"{args.tag} -> {v.label}")
        elif args.cmd == "run":
            r = _run(reg, args, args.ref)
            print(f"{r.prompt} v{r.version}: {r.passed}/{r.n} passed ({r.pass_rate:.1%}), mean score {r.mean_score:.3f}")
            for x in r.results:
                if args.verbose or not x.passed:
                    tag = f" [{x.pass_count}/{x.samples}{' flaky' if x.flaky else ''}]" if x.samples > 1 else ""
                    print(f"  {'PASS' if x.passed else 'FAIL'} {x.id}{tag}: {x.error or '; '.join(x.failures) or 'ok'}")
            if r.flaky:
                print(f"flaky cases ({len(r.flaky)}): {', '.join(r.flaky)}")
            if args.out:
                r.save(args.out)
        elif args.cmd == "compare":
            ra, rb = _run(reg, args, args.a), _run(reg, args, args.b)
            c = compare_runs(ra, rb, alpha=args.alpha)
            md = render_markdown(c)
            print(md)
            if args.report:
                Path(args.report).write_text(md, encoding="utf-8")
        elif args.cmd == "baseline":
            r = _run(reg, args, args.ref)
            r.save(args.out)
            print(f"baseline saved: {r.prompt} v{r.version}, {r.passed}/{r.n} passed -> {args.out}")
        elif args.cmd == "calibrate":
            judge = make_judge(args.judge)
            cal = calibrate_judge(load_calibration_items(args.labels), judge, args.min_score, args.votes)
            print(render_calibration(cal))
            if args.min_kappa is not None and (cal.kappa is None or cal.kappa < args.min_kappa):
                print(f"FAIL: kappa {'undefined' if cal.kappa is None else f'{cal.kappa:.2f}'} < required {args.min_kappa}")
                return 1
        elif args.cmd == "pairwise":
            cases = load_cases(args.cases)
            outs = []
            for ref in (args.a, args.b):
                r = run_prompt(reg.resolve(args.name, ref), cases, make_model(args.model))
                outs.append(r.by_id())
            items = []
            for c in cases:
                rubric = args.rubric or next((ch.get("rubric") for ch in c.checks
                                              if ch.get("type") == "judge" and ch.get("rubric")), None)
                if not rubric:
                    raise ValueError(f"case {c.id!r} has no judge rubric: pass --rubric")
                items.append({"id": c.id, "rubric": rubric, "vars": c.vars,
                              "a": outs[0][c.id].output, "b": outs[1][c.id].output})
            summ = compare_outputs(make_pair_judge(args.judge), items, args.alpha)
            print(render_pairwise(summ, f"{args.name}@{args.a}", f"{args.name}@{args.b}"))
            if args.out:
                import json
                Path(args.out).write_text(json.dumps([v.to_dict() for v in summ.verdicts], indent=2) + "\n", encoding="utf-8")
            if summ.errors:
                print(f"FAIL: {summ.errors} pair(s) could not be judged")
                return 1
            if args.max_inconsistent is not None and summ.inconsistent_rate > args.max_inconsistent:
                print(f"FAIL: {summ.inconsistent_rate:.1%} of pairs flip with order, above {args.max_inconsistent:.1%}")
                return 1
        elif args.cmd == "pair-audit":
            a = audit_judge(load_pair_items(args.labels), make_pair_judge(args.judge))
            print(render_audit(a))
            if args.min_accuracy is not None and a.reconciled_accuracy < args.min_accuracy:
                print(f"FAIL: reconciled accuracy {a.reconciled_accuracy:.1%} < required {args.min_accuracy:.1%}")
                return 1
            if args.max_inconsistent is not None and 1 - a.consistency > args.max_inconsistent:
                print(f"FAIL: {1 - a.consistency:.1%} of pairs flip with order, above {args.max_inconsistent:.1%}")
                return 1
        elif args.cmd == "check":
            cur = _run(reg, args, args.ref)
            rep = check_regression(cur, RunResult.load(args.baseline), args.max_regressions, args.min_pass_rate, args.max_drop)
            print(f"{'OK' if rep.ok else 'FAIL'}: pass rate {rep.pass_rate:.1%} (baseline {rep.baseline_pass_rate:.1%}); "
                  f"{len(rep.regressions)} regressed, {len(rep.fixed)} fixed, {len(rep.new_cases)} new")
            for r in rep.reasons:
                print(f"  - {r}")
            if args.summary:
                with open(args.summary, "a", encoding="utf-8") as f:
                    f.write(render_check_markdown(rep))
            return 0 if rep.ok else 1
    except (RegistryError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
