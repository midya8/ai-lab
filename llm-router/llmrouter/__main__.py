"""CLI: ``python -m llmrouter {explain,simulate} ...``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .budget import Budget
from .models import CatalogueError, load_models
from .providers import SimProvider
from .router import STRATEGIES, Router, RouterError
from .sim import compare, format_table, parse_outage, simulate

DEFAULT_MODELS = Path(__file__).resolve().parent.parent / "examples" / "models.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llmrouter", description=__doc__)
    p.add_argument("--models", default=str(DEFAULT_MODELS), help="model catalogue JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explain", help="show which model a prompt would be routed to (no calls)")
    e.add_argument("prompt")
    e.add_argument("--strategy", choices=STRATEGIES, default="balanced")
    e.add_argument("--max-tokens", type=int, default=256)
    e.add_argument("--json", action="store_true")

    s = sub.add_parser("simulate", help="replay a synthetic workload and compare strategies")
    s.add_argument("--requests", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--strategy", choices=STRATEGIES, help="one strategy (default: compare all)")
    s.add_argument("--outage", action="append", default=[], metavar="MODEL:START:END",
                   help="take a model down for a request-index range (repeatable)")
    s.add_argument("--budget", type=float, help="USD budget for the whole run")
    s.add_argument("--failure-rate", type=float, default=0.02)
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    try:
        models = load_models(args.models)
    except (OSError, ValueError, CatalogueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "explain":
        router = Router(models, SimProvider(), strategy=args.strategy)
        try:
            info = router.explain(args.prompt, args.max_tokens)
        except RouterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"complexity {info['complexity']:.2f} -> needs quality >= {info['required_quality']:.2f}")
            for r in info["reasons"] or ["no difficulty cues"]:
                print(f"  - {r}")
            for i, c in enumerate(info["candidates"]):
                print(f"  {'->' if i == 0 else '  '} {c['model']:<9} ~${c['est_cost_usd']:.6f}  ~{c['est_latency_ms']} ms")
        return 0

    try:
        outages = [parse_outage(o) for o in args.outage]
        kw = dict(n=args.requests, seed=args.seed, outages=outages, budget_usd=args.budget,
                  failure_rate=args.failure_rate, use_cache=not args.no_cache)
        rows = [simulate(models, args.strategy, **kw)] if args.strategy else compare(models, **kw)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_table(rows))
        print("\nbaseline = every request sent to the highest-quality model; "
              "usage by model:")
        for r in rows:
            print(f"  {r['strategy']:<9}", r["per_model"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
