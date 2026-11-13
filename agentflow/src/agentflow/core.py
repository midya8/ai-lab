"""Multi-agent orchestration: agents with tools run tasks arranged in a dependency DAG.

Agents follow a ReAct-style loop. The LLM returns either
  {"tool": name, "args": {...}}  -> tool call, result fed back as an observation
  {"final": text}                -> task answer
Independent tasks run in parallel threads; failures retry with backoff and
cascade as "skipped" to dependents.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


class CycleError(ValueError):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def call(self, name: str, args: dict):
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name](**args)

    def names(self) -> List[str]:
        return sorted(self._tools)


class ScriptedLLM:
    """Deterministic LLM stand-in: replays queued replies (str or dict) per call.
    Swap in any object with .complete(messages) -> str to use a real model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages: List[dict]) -> str:
        if self.calls >= len(self.replies):
            raise RuntimeError("ScriptedLLM exhausted")
        r = self.replies[self.calls]
        self.calls += 1
        return r if isinstance(r, str) else json.dumps(r)


@dataclass
class Agent:
    name: str
    llm: object
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    system: str = ""
    max_steps: int = 6

    def run(self, prompt: str, scope=None) -> str:
        """``scope`` (a ToolScope) replays journaled tool calls instead of re-running them."""
        messages = [{"role": "system", "content": self.system or f"You are {self.name}. Tools: {self.tools.names()}"},
                    {"role": "user", "content": prompt}]
        for _ in range(self.max_steps):
            raw = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": raw})
            try:
                msg = json.loads(raw)
                if not isinstance(msg, dict):
                    raise ValueError
            except ValueError:
                return raw  # plain text is a final answer
            if "final" in msg:
                return str(msg["final"])
            if "tool" in msg:
                try:
                    if scope is None:
                        obs = self.tools.call(msg["tool"], msg.get("args", {}))
                    else:
                        obs = scope.call_sync(msg["tool"], msg.get("args", {}),
                                              lambda t, a: self.tools.call(t, a))
                except Exception as e:  # surfaced to the model so it can recover
                    obs = f"ERROR: {e}"
                messages.append({"role": "user", "content": f"observation: {obs}"})
            else:
                return raw
        raise RuntimeError(f"{self.name} exceeded {self.max_steps} steps")


@dataclass
class Task:
    id: str
    agent: str
    prompt: str  # may reference upstream outputs as {dep_id}
    deps: List[str] = field(default_factory=list)
    retries: int = 0


@dataclass
class Result:
    status: str  # ok | failed | skipped
    output: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    duration: float = 0.0  # wall seconds spent executing (incl. retries)
    resumed: bool = False  # replayed from a RunLog instead of executed


class _Safe(dict):
    def __missing__(self, k):
        return "{" + k + "}"


class Orchestrator:
    def __init__(self, agents: List[Agent], max_workers: int = 4, backoff: float = 0.0, tool_journal=None):
        self.tool_journal = tool_journal  # ToolJournal or None
        self.tool_stats = {"replayed": 0, "live": 0}
        self._fps: Dict[str, str] = {}
        self._stats_lock = threading.Lock()
        self.agents = {a.name: a for a in agents}
        self.max_workers = max_workers
        self.backoff = backoff

    def plan(self, tasks: List[Task]) -> List[List[str]]:
        """Topological layers; raises CycleError / KeyError on bad graphs."""
        ids = {t.id: t for t in tasks}
        for t in tasks:
            if t.agent not in self.agents:
                raise KeyError(f"task {t.id}: unknown agent {t.agent}")
            for d in t.deps:
                if d not in ids:
                    raise KeyError(f"task {t.id}: unknown dependency {d}")
        done, layers, remaining = set(), [], dict(ids)
        while remaining:
            layer = sorted(i for i, t in remaining.items() if all(d in done for d in t.deps))
            if not layer:
                raise CycleError(f"cycle among: {sorted(remaining)}")
            layers.append(layer)
            done.update(layer)
            for i in layer:
                del remaining[i]
        return layers

    def _exec(self, task: Task, outputs: Dict[str, str]) -> Result:
        t0 = time.perf_counter()
        res = self._exec_inner(task, outputs)
        res.duration = time.perf_counter() - t0
        return res

    def _exec_inner(self, task: Task, outputs: Dict[str, str]) -> Result:
        prompt = task.prompt.format_map(_Safe(outputs))
        err = None
        for attempt in range(1, task.retries + 2):
            try:
                scope = self.tool_journal.scope(self._fps[task.id]) if self.tool_journal is not None else None
                try:
                    out = self.agents[task.agent].run(prompt, scope) if scope is not None else self.agents[task.agent].run(prompt)
                finally:
                    if scope is not None:
                        with self._stats_lock:
                            self.tool_stats["replayed"] += scope.replayed
                            self.tool_stats["live"] += scope.live
                return Result("ok", out, attempts=attempt)
            except Exception as e:
                err = str(e)
                if attempt <= task.retries:
                    time.sleep(self.backoff * 2 ** (attempt - 1))
        return Result("failed", error=err, attempts=task.retries + 1)

    def run(self, tasks: List[Task], log=None, resume: bool = False,
            run_id: Optional[str] = None) -> Dict[str, Result]:
        """Run the DAG. With ``log`` (a RunLog) every result is journaled as it lands;
        with ``resume=True`` tasks already completed in the log (same fingerprint)
        are replayed instead of re-executed."""
        self.plan(tasks)
        fps = {}
        if log is not None or self.tool_journal is not None:
            from .runlog import fingerprints
            fps = fingerprints(tasks)
            self._fps = fps
        if log is not None:
            run_id = run_id or log.new_run_id()
        pending = {t.id: t for t in tasks}
        results: Dict[str, Result] = {}
        if log is not None and resume:
            for tid, e in log.completed(tasks).items():
                results[tid] = Result("ok", e["output"], attempts=0, duration=0.0, resumed=True)
                log.record(run_id, tid, fps[tid], results[tid], resumed=True)
                del pending[tid]
        running = {}
        with ThreadPoolExecutor(self.max_workers) as pool:
            while pending or running:
                for tid, t in list(pending.items()):
                    if any(d in results and results[d].status != "ok" for d in t.deps):
                        results[tid] = Result("skipped", error="upstream failure")
                        if log is not None:
                            log.record(run_id, tid, fps[tid], results[tid])
                        del pending[tid]
                    elif all(d in results for d in t.deps):
                        outs = {k: r.output for k, r in results.items() if r.output is not None}
                        running[pool.submit(self._exec, t, outs)] = tid
                        del pending[tid]
                if running:
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for f in finished:
                        tid = running.pop(f)
                        results[tid] = f.result()
                        if log is not None:
                            log.record(run_id, tid, fps[tid], results[tid])
        return results
