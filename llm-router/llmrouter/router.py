"""The router: pick a model per prompt, call it, recover from failure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .breaker import CircuitBreaker
from .budget import Budget, BudgetExceeded
from .cache import ResponseCache, cache_key
from .semantic import SemanticCache
from .clock import SystemClock
from .complexity import Complexity, score_prompt
from .latency import LatencyTracker
from .models import ModelSpec, estimate_tokens, expected_output_tokens
from .providers import Provider, ProviderError

STRATEGIES = ("cheapest", "fastest", "balanced", "quality")


class RouterError(RuntimeError):
    """The request cannot be routed (e.g. prompt exceeds every context window)."""


class AllModelsFailed(RouterError):
    def __init__(self, attempts: list["Attempt"]) -> None:
        self.attempts = attempts
        detail = "; ".join(f"{a.model}: {a.note}" for a in attempts) or "no candidates"
        super().__init__(f"all candidate models failed or were unavailable ({detail})")


@dataclass
class Attempt:
    model: str
    outcome: str          # ok | error | skipped | rejected
    note: str = ""
    cost: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Plan:
    candidates: list[ModelSpec]
    complexity: Complexity
    required_quality: float
    strategy: str
    degraded: bool
    input_tokens: int
    expected_output_tokens: int


@dataclass
class RouteResult:
    text: str
    model: str
    strategy: str
    complexity: Complexity
    required_quality: float
    cached: bool
    cost: float
    latency_ms: float
    attempts: list[Attempt] = field(default_factory=list)
    verified: bool | None = None
    degraded: bool = False

    @property
    def fallbacks(self) -> int:
        return sum(1 for a in self.attempts if a.outcome in ("error", "rejected"))


class Router:
    def __init__(
        self,
        models: list[ModelSpec],
        provider: Provider,
        strategy: str = "balanced",
        cache: "ResponseCache | SemanticCache | None" = None,
        budget: Budget | None = None,
        clock=None,
        breaker_threshold: int = 3,
        breaker_cooldown_s: float = 30.0,
        degrade_quality: float = 0.2,
        balance_weights: tuple[float, float] = (0.6, 0.4),
        allow_downgrade_on_failure: bool = False,
        latency_tracker: "LatencyTracker | None" = None,
        scorer=None,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        self.models = list(models)
        self.provider = provider
        self.strategy = strategy
        self.cache = cache
        self.budget = budget
        self.clock = clock or SystemClock()
        self.degrade_quality = degrade_quality
        self.balance_weights = balance_weights
        self.allow_downgrade = allow_downgrade_on_failure
        self.latency_tracker = latency_tracker
        self.scorer = scorer or score_prompt
        self.breakers = {
            m.name: CircuitBreaker(breaker_threshold, breaker_cooldown_s, self.clock)
            for m in self.models
        }
        self._top = max(self.models, key=lambda m: m.quality)
        self._stats = {
            "requests": 0, "cache_hits": 0, "cost": 0.0, "baseline_cost": 0.0,
            "latency_ms": 0.0, "fallbacks": 0, "failures": 0, "degraded": 0,
            "per_model": {m.name: 0 for m in self.models},
        }

    # ------------------------------------------------------------ planning
    def plan(self, prompt: str, max_tokens: int = 256, strategy: str | None = None,
             cascade: bool = False) -> Plan:
        strategy = strategy or self.strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        complexity = self.scorer(prompt)
        required = complexity.required_quality()
        degraded = bool(self.budget and self.budget.degraded)
        if degraded:
            required = max(0.0, required - self.degrade_quality)
            strategy = "cheapest"

        in_tok = estimate_tokens(prompt)
        out_tok = expected_output_tokens(prompt, max_tokens)
        fits = [m for m in self.models if m.context_window >= in_tok]
        if not fits:
            raise RouterError(
                f"prompt (~{in_tok} tokens) exceeds every model's context window")
        required = min(required, max(m.quality for m in fits))

        if cascade:
            # Cheap-first ladder: the verifier, not the heuristic, guards quality.
            ordered = sorted(fits, key=lambda m: (m.cost(in_tok, out_tok), m.name))
        else:
            eligible = [m for m in fits if m.quality >= required - 1e-9]
            ordered = self._order(eligible, strategy, in_tok, out_tok)
            if self.allow_downgrade:
                rest = sorted((m for m in fits if m not in eligible),
                              key=lambda m: (-m.quality, m.name))
                ordered += rest
        return Plan(ordered, complexity, round(required, 4), strategy, degraded,
                    in_tok, out_tok)

    def _lat(self, m, out_tok: int) -> float:
        """Predicted latency, corrected by observed behaviour when tracked."""
        base = m.latency(out_tok)
        return base * self.latency_tracker.factor(m.name) if self.latency_tracker else base

    def _order(self, models, strategy, in_tok, out_tok):
        if strategy == "cheapest":
            key = lambda m: (m.cost(in_tok, out_tok), self._lat(m, out_tok), m.name)
        elif strategy == "fastest":
            key = lambda m: (self._lat(m, out_tok), m.cost(in_tok, out_tok), m.name)
        elif strategy == "quality":
            key = lambda m: (-m.quality, m.cost(in_tok, out_tok), m.name)
        else:  # balanced: min-max normalised weighted sum of cost and latency
            costs = {m.name: m.cost(in_tok, out_tok) for m in models}
            lats = {m.name: self._lat(m, out_tok) for m in models}
            mc, ml = max(costs.values(), default=0) or 1.0, max(lats.values(), default=0) or 1.0
            wc, wl = self.balance_weights
            key = lambda m: (wc * costs[m.name] / mc + wl * lats[m.name] / ml, m.name)
        return sorted(models, key=key)

    def explain(self, prompt: str, max_tokens: int = 256, strategy: str | None = None) -> dict:
        """Routing decision without calling any model."""
        p = self.plan(prompt, max_tokens, strategy)
        return {
            "strategy": p.strategy,
            "complexity": p.complexity.score,
            "reasons": list(p.complexity.reasons),
            "required_quality": p.required_quality,
            "degraded_by_budget": p.degraded,
            "candidates": [
                {"model": m.name, "est_cost_usd": round(m.cost(p.input_tokens, p.expected_output_tokens), 6),
                 "est_latency_ms": round(m.latency(p.expected_output_tokens)),
                 "breaker": self.breakers[m.name].state}
                for m in p.candidates
            ],
        }

    # ------------------------------------------------------------- routing
    def route(self, prompt: str, max_tokens: int = 256, strategy: str | None = None,
              verifier: Callable[[str], bool] | None = None) -> RouteResult:
        """Route one prompt.

        With a ``verifier`` the router runs a cascade: cheapest model first,
        escalating to the next-cheapest while the verifier rejects the answer.
        """
        plan = self.plan(prompt, max_tokens, strategy, cascade=verifier is not None)
        self._stats["requests"] += 1
        if plan.degraded:
            self._stats["degraded"] += 1

        key = cache_key(prompt, max_tokens)
        if self.cache is not None and verifier is None:
            if hasattr(self.cache, "lookup"):
                found = self.cache.lookup(prompt, max_tokens)
                hit = found[0] if found else None
            else:
                hit = self.cache.get(key)
            if hit is not None:
                self._stats["cache_hits"] += 1
                self._stats["baseline_cost"] += self._top.cost(hit["in"], hit["out"])
                return RouteResult(hit["text"], hit["model"], plan.strategy, plan.complexity,
                                   plan.required_quality, True, 0.0, 0.0, [], None, plan.degraded)

        attempts: list[Attempt] = []
        best: tuple | None = None  # (completion, spec, verified)
        total_cost = total_latency = 0.0
        budget_blocked = False

        for spec in plan.candidates:
            est = spec.cost(plan.input_tokens, plan.expected_output_tokens)
            if self.budget is not None and not self.budget.can_afford(est):
                budget_blocked = True
                attempts.append(Attempt(spec.name, "skipped", f"over budget (needs ~${est:.5f})"))
                continue
            if not self.breakers[spec.name].allow():
                attempts.append(Attempt(spec.name, "skipped", "circuit open"))
                continue
            try:
                comp = self.provider.complete(spec, prompt, max_tokens)
            except ProviderError as exc:
                self.breakers[spec.name].record_failure()
                self._stats["failures"] += 1
                attempts.append(Attempt(spec.name, "error", str(exc)))
                continue
            self.breakers[spec.name].record_success()
            if self.latency_tracker is not None:
                self.latency_tracker.record(spec.name, spec.latency(comp.output_tokens),
                                            comp.latency_ms)
            cost = spec.cost(comp.input_tokens, comp.output_tokens)
            if self.budget is not None:
                self.budget.record(cost)
            total_cost += cost
            total_latency += comp.latency_ms
            if verifier is not None and not verifier(comp.text):
                attempts.append(Attempt(spec.name, "rejected", "verifier rejected answer",
                                        cost, comp.latency_ms))
                best = (comp, spec, False)
                continue
            attempts.append(Attempt(spec.name, "ok", "", cost, comp.latency_ms))
            best = (comp, spec, True if verifier is not None else None)
            break

        if best is None:
            if budget_blocked and all(a.outcome == "skipped" for a in attempts):
                raise BudgetExceeded("no candidate model fits in the remaining budget")
            raise AllModelsFailed(attempts)

        comp, spec, verified = best
        fallbacks = sum(1 for a in attempts if a.outcome in ("error", "rejected"))
        self._stats["cost"] += total_cost
        self._stats["latency_ms"] += total_latency
        self._stats["fallbacks"] += fallbacks
        self._stats["per_model"][spec.name] += 1
        self._stats["baseline_cost"] += self._top.cost(comp.input_tokens, comp.output_tokens)
        if self.cache is not None and verified is not False:
            entry = {"text": comp.text, "model": spec.name,
                     "in": comp.input_tokens, "out": comp.output_tokens}
            if hasattr(self.cache, "store"):
                self.cache.store(prompt, max_tokens, entry)
            else:
                self.cache.put(key, entry)
        return RouteResult(comp.text, spec.name, plan.strategy, plan.complexity,
                           plan.required_quality, False, total_cost, total_latency,
                           attempts, verified, plan.degraded)

    # --------------------------------------------------------------- stats
    def stats(self) -> dict:
        s = self._stats
        n = s["requests"]
        live = n - s["cache_hits"]
        saved = s["baseline_cost"] - s["cost"]
        return {
            "requests": n,
            "cache_hits": s["cache_hits"],
            "total_cost_usd": round(s["cost"], 6),
            "baseline_cost_usd": round(s["baseline_cost"], 6),
            "savings_pct": round(100 * saved / s["baseline_cost"], 1) if s["baseline_cost"] else 0.0,
            "avg_latency_ms": round(s["latency_ms"] / live, 1) if live else 0.0,
            "fallbacks": s["fallbacks"],
            "provider_failures": s["failures"],
            "budget_degraded_requests": s["degraded"],
            "per_model": dict(s["per_model"]),
            **({"observed_latency": self.latency_tracker.snapshot()} if self.latency_tracker else {}),
        }
