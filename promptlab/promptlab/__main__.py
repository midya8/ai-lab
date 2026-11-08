from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .compare import check_regression, render_check_markdown, compare_runs, render_markdown
from .models import make_model
from .registry import Registry, RegistryError
from .runner import RunResult, load_cases, run_prompt


def _run(reg: Registry, args, ref: str) -> RunResult:
    v = reg.resolve(args.name, ref)
    return run_prompt(v, load_cases(args.cases), make_model(args.model),
                      samples=args.samples, threshold=args.threshold)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promptlab", description="Prompt versioning, A/B testing and regression gates")
    ap.add_argument("--root", default=os.environ.get("PROMPTLAB_ROOT", "prompts"), help="registry directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_run_opts(p):
        p.add_argument("--cases", required=True, help="JSONL test cases")
        p.add_argument("--model", default="rule", help="rule | rule:<seed> | cmd:<command>")
        p.add_argument("--samples", type=int, default=1, help="model calls per case (non-deterministic models)")
        p.add_argument("--threshold", type=float, default=1.0, help="fraction of samples that must pass a case")

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
