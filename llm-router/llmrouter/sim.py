"""Workload simulator: replay a realistic prompt mix through the router."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .budget import Budget, BudgetExceeded
from .cache import ResponseCache
from .clock import ManualClock
from .models import ModelSpec
from .providers import SimProvider
from .router import STRATEGIES, Router, RouterError

_SIMPLE = [
    "Hi there!", "What is the capital of France?", "Define photosynthesis.",
    "Translate 'good morning' to Spanish.", "Who is Ada Lovelace?",
    "What's the boiling point of water in Celsius?", "Thanks, that helps!",
]
_MEDIUM = [
    "Summarize the main causes of the French Revolution in a short paragraph.",
    "Write a polite email asking my landlord to fix the heating. Keep it under 120 words.",
    "Explain the difference between a process and a thread.",
    "Give me five names for a coffee shop that sells books.",
]
_HARD = [
    "Analyze the trade-offs between event sourcing and CRUD for a banking ledger. "
    "You must cover consistency, auditability and operational cost, and do not recommend a vendor.",
    "Debug this function and explain why it fails:\n```python\ndef avg(xs):\n    return sum(xs) / len(xs)\n```\n"
    "1. Find the root cause\n2. Propose a fix\n3. Write tests",
    "Prove that the sum of two even integers is even, step by step, then derive a general result "
    "for the sum of n even integers.",
    "Design a rate limiter for a multi-region API. Compare and contrast token bucket and sliding "
    "window, ensure that it handles clock skew, and optimize for p99 latency.",
]


def build_workload(n: int, seed: int = 0, repeat_rate: float = 0.10) -> list[str]:
    """Mixed workload: ~50% simple, ~30% medium, ~20% hard, plus verbatim repeats."""
    rng = random.Random(seed)
    prompts: list[str] = []
    for i in range(n):
        if prompts and rng.random() < repeat_rate:
            prompts.append(rng.choice(prompts))
            continue
        r = rng.random()
        pool = _SIMPLE if r < 0.5 else _MEDIUM if r < 0.8 else _HARD
        base = rng.choice(pool)
        prompts.append(f"{base} (ref #{i})" if rng.random() < 0.8 else base)
    return prompts


@dataclass(frozen=True)
class Outage:
    model: str
    start: int  # request index (inclusive)
    end: int    # request index (exclusive)


def parse_outage(spec: str) -> Outage:
    try:
        model, start, end = spec.rsplit(":", 2)
        out = Outage(model, int(start), int(end))
    except ValueError as exc:
        raise ValueError(f"outage must look like MODEL:START:END, got {spec!r}") from exc
    if out.start < 0 or out.end <= out.start:
        raise ValueError("outage needs 0 <= START < END")
    return out


def simulate(models: list[ModelSpec], strategy: str, n: int = 300, seed: int = 0,
             outages: list[Outage] | None = None, budget_usd: float | None = None,
             failure_rate: float = 0.02, use_cache: bool = True) -> dict:
    clock = ManualClock()
    provider = SimProvider({m.name: failure_rate for m in models}, seed=seed, clock=clock)
    router = Router(
        models, provider, strategy=strategy,
        cache=ResponseCache(clock=clock) if use_cache else None,
        budget=Budget(budget_usd, clock=clock) if budget_usd else None,
        clock=clock,
    )
    rejected = errors = 0
    for i, prompt in enumerate(build_workload(n, seed)):
        provider.down = {o.model for o in (outages or []) if o.start <= i < o.end}
        try:
            router.route(prompt)
        except BudgetExceeded:
            rejected += 1
        except RouterError:
            errors += 1
        clock.advance(1.0)
    out = router.stats()
    out.update(strategy=strategy, rejected_over_budget=rejected, unroutable=errors)
    return out


def compare(models: list[ModelSpec], **kw) -> list[dict]:
    return [simulate(models, s, **kw) for s in STRATEGIES]


def format_table(rows: list[dict]) -> str:
    head = f"{'strategy':<10} {'cost $':>9} {'baseline $':>11} {'saved':>7} {'avg ms':>8} {'cache':>6} {'fallbk':>7} {'dropped':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['strategy']:<10} {r['total_cost_usd']:>9.4f} {r['baseline_cost_usd']:>11.4f} "
            f"{r['savings_pct']:>6.1f}% {r['avg_latency_ms']:>8.0f} {r['cache_hits']:>6} "
            f"{r['fallbacks']:>7} {r['rejected_over_budget'] + r['unroutable']:>8}")
    return "\n".join(lines)
