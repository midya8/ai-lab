"""Tracing and supervisor routing for agentflow.

TracedLLM / TracedTools wrap an agent's LLM and tools and record spans
(name, kind, duration, status) into a shared Tracer, exportable as JSON.
Supervisor routes a request to the best-fit worker agent via an LLM that
replies {"route": name}; falls back to a default on an invalid choice.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

from .core import Agent, ToolRegistry


@dataclass
class Span:
    agent: str
    kind: str  # llm | tool
    name: str
    duration_ms: float
    status: str = "ok"
    error: Optional[str] = None


class Tracer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.spans: List[Span] = []
        self._lock = threading.Lock()
        self._clock = clock

    def record(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    def summary(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for s in self.spans:
            d = out.setdefault(s.agent, {"llm_calls": 0, "tool_calls": 0, "errors": 0, "total_ms": 0.0})
            d["llm_calls" if s.kind == "llm" else "tool_calls"] += 1
            d["errors"] += s.status != "ok"
            d["total_ms"] = round(d["total_ms"] + s.duration_ms, 3)
        return out

    def to_json(self) -> str:
        return json.dumps([asdict(s) for s in self.spans], indent=2)


class _TracedLLM:
    def __init__(self, inner, agent, tracer):
        self.inner, self.agent, self.tracer = inner, agent, tracer

    def complete(self, messages):
        t = self.tracer._clock()
        try:
            r = self.inner.complete(messages)
        except Exception as e:
            self.tracer.record(Span(self.agent, "llm", "complete", (self.tracer._clock() - t) * 1e3, "error", str(e)))
            raise
        self.tracer.record(Span(self.agent, "llm", "complete", (self.tracer._clock() - t) * 1e3))
        return r


class _TracedTools(ToolRegistry):
    def __init__(self, inner: ToolRegistry, agent, tracer):
        self._tools = inner._tools
        self.agent, self.tracer = agent, tracer

    def call(self, name, args):
        t = self.tracer._clock()
        try:
            r = ToolRegistry.call(self, name, args)
        except Exception as e:
            self.tracer.record(Span(self.agent, "tool", name, (self.tracer._clock() - t) * 1e3, "error", str(e)))
            raise
        self.tracer.record(Span(self.agent, "tool", name, (self.tracer._clock() - t) * 1e3))
        return r


def trace_agent(agent: Agent, tracer: Tracer) -> Agent:
    """Instrument an agent in place and return it."""
    agent.llm = _TracedLLM(agent.llm, agent.name, tracer)
    agent.tools = _TracedTools(agent.tools, agent.name, tracer)
    return agent


class Supervisor:
    def __init__(self, router_llm, workers: List[Agent], default: Optional[str] = None,
                 memory=None, recall_limit: int = 3, remember_results: bool = True,
                 max_attempts: int = 1):
        if not workers:
            raise ValueError("need at least one worker")
        self.router_llm = router_llm
        self.workers = {w.name: w for w in workers}
        self.default = default or workers[0].name
        self.memory = memory
        self.recall_limit = recall_limit
        self.remember_results = remember_results
        self.max_attempts = max(1, max_attempts)
        if self.default not in self.workers:
            raise KeyError(f"unknown default agent {self.default}")

    def route(self, request: str) -> str:
        roster = "; ".join(f"{n}: {w.system or 'general'}" for n, w in self.workers.items())
        raw = self.router_llm.complete([
            {"role": "system", "content": f'Pick the best worker. Reply JSON {{"route": name}}. Workers: {roster}'},
            {"role": "user", "content": request}])
        try:
            choice = json.loads(raw).get("route") if raw.strip().startswith("{") else raw.strip()
        except (ValueError, AttributeError):
            choice = raw.strip()
        return choice if choice in self.workers else self.default

    def _context(self, request: str) -> list:
        if self.memory is None:
            return []
        return self.memory.search(request, self.recall_limit)

    def run(self, request: str) -> dict:
        name = self.route(request)
        ctx = self._context(request)
        prompt = request
        if ctx:
            notes = "\n".join(f"- {e['key']}: {e['value']}" for e in ctx)
            prompt = f"Relevant shared memory:\n{notes}\n\nRequest: {request}"
        order = [name] + [n for n in self.workers if n != name]
        failures = []
        out = None
        for attempt, cand in enumerate(order[:self.max_attempts], 1):
            p = prompt
            if failures:
                p += "\n\nEarlier attempts failed:\n" + "\n".join(f"- {a}: {e}" for a, e in failures)
            try:
                out = self.workers[cand].run(p)
                name = cand
                break
            except Exception as e:  # fall through to the next worker
                failures.append((cand, str(e)))
                if self.memory is not None:
                    self.memory.remember(f"failure:{cand}:{request[:40]}", str(e), author="supervisor")
        else:
            raise RuntimeError("all attempts failed: " + "; ".join(f"{a}: {e}" for a, e in failures))
        if self.memory is not None and self.remember_results:
            self.memory.remember(f"supervisor:{request[:60]}", out, author=name)
        result = {"agent": name, "output": out}
        if failures:
            result["failed_attempts"] = [a for a, _ in failures]
        if self.memory is not None:
            result["memory_used"] = [e["key"] for e in ctx]
        return result
