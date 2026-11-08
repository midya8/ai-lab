"""Run agentflow agents on top of an llm-router ``Router``.

agentflow stays dependency-free: the adapter is duck-typed against the small
slice of the router API it needs (``route(prompt, max_tokens, strategy,
verifier)`` returning an object with ``text``, ``model``, ``cost`` ...). Any
object with that shape works, including ``llmrouter.Router``.

Typical use::

    backend = RouterBackend(router)
    planner = Agent("planner", backend.llm("planner", strategy="quality"))
    scraper = Agent("scraper", backend.llm("scraper", strategy="cheapest"))
    ...
    print(backend.report())      # cost / model mix per agent

Agents running in parallel threads share one router, so calls are serialized
through a lock owned by the backend (the router keeps unsynchronised stats).
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional


def render_messages(messages: List[dict]) -> str:
    """Flatten chat messages into one prompt string (stable, so cache-friendly)."""
    return "\n".join(f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages)


class RouterBackend:
    """Shared state for every ``RouterLLM`` built from one router."""

    def __init__(self, router, max_cost: Optional[float] = None):
        self.router = router
        self.max_cost = max_cost
        self._lock = threading.Lock()
        self.records: List[dict] = []

    # ------------------------------------------------------------ factory
    def llm(self, name: str, strategy: Optional[str] = None, max_tokens: int = 256,
            verifier: Optional[Callable[[str], bool]] = None) -> "RouterLLM":
        return RouterLLM(self, name, strategy, max_tokens, verifier)

    # ------------------------------------------------------------- calls
    @property
    def total_cost(self) -> float:
        return sum(r["cost"] for r in self.records)

    def _call(self, name, prompt, strategy, max_tokens, verifier) -> str:
        with self._lock:
            if self.max_cost is not None and self.total_cost >= self.max_cost:
                raise BudgetExhausted(
                    f"agent budget ${self.max_cost:.6f} reached (spent ${self.total_cost:.6f})")
            res = self.router.route(prompt, max_tokens=max_tokens, strategy=strategy,
                                    verifier=verifier)
            self.records.append({
                "agent": name, "model": res.model, "strategy": res.strategy,
                "cost": float(res.cost), "latency_ms": float(res.latency_ms),
                "cached": bool(res.cached), "fallbacks": int(getattr(res, "fallbacks", 0)),
                "verified": getattr(res, "verified", None),
            })
        return res.text

    # ------------------------------------------------------------ reports
    def report(self) -> dict:
        """Aggregate usage: totals, per agent, and per model."""
        def blank():
            return {"calls": 0, "cost": 0.0, "latency_ms": 0.0, "cache_hits": 0, "fallbacks": 0}
        total, per_agent, per_model = blank(), {}, {}
        for r in self.records:
            for bucket in (total, per_agent.setdefault(r["agent"], blank()),
                           per_model.setdefault(r["model"], blank())):
                bucket["calls"] += 1
                bucket["cost"] += r["cost"]
                bucket["latency_ms"] += r["latency_ms"]
                bucket["cache_hits"] += r["cached"]
                bucket["fallbacks"] += r["fallbacks"]
        for b in [total, *per_agent.values(), *per_model.values()]:
            b["cost"] = round(b["cost"], 8)
            b["latency_ms"] = round(b["latency_ms"], 2)
        return {"total": total, "per_agent": per_agent, "per_model": per_model}

    def format_report(self) -> str:
        rep = self.report()
        lines = ["agent        calls  cost($)     models",
                 "-----------  -----  ----------  ------"]
        for agent, b in sorted(rep["per_agent"].items()):
            models = sorted({r["model"] for r in self.records if r["agent"] == agent})
            lines.append(f"{agent:<11}  {b['calls']:>5}  {b['cost']:>10.6f}  {', '.join(models)}")
        t = rep["total"]
        lines.append(f"{'TOTAL':<11}  {t['calls']:>5}  {t['cost']:>10.6f}  "
                     f"cache hits {t['cache_hits']}, fallbacks {t['fallbacks']}")
        return "\n".join(lines)


class BudgetExhausted(RuntimeError):
    """The backend's spend cap was reached; further LLM calls are refused."""


class RouterLLM:
    """An agentflow LLM (``.complete(messages) -> str``) backed by the router."""

    def __init__(self, backend: RouterBackend, name: str, strategy: Optional[str] = None,
                 max_tokens: int = 256, verifier: Optional[Callable[[str], bool]] = None):
        self.backend, self.name = backend, name
        self.strategy, self.max_tokens, self.verifier = strategy, max_tokens, verifier

    def complete(self, messages: List[dict]) -> str:
        return self.backend._call(self.name, render_messages(messages), self.strategy,
                                  self.max_tokens, self.verifier)


def json_or_text(text: str) -> bool:
    """Verifier for the agent protocol: a JSON object must carry ``final`` or ``tool``.

    Plain text is accepted (agentflow treats it as a final answer); malformed
    protocol messages such as ``{"oops": 1}`` are rejected so a router cascade
    can escalate to a stronger model.
    """
    import json
    try:
        msg = json.loads(text)
    except ValueError:
        return True
    if not isinstance(msg, dict):
        return True
    return "final" in msg or "tool" in msg
