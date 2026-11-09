"""asyncio orchestration: the same Task DAG as ``Orchestrator``, on an event loop.

Why: LLM calls are I/O bound, so thousands of in-flight tasks cost a semaphore
rather than a thread each. Agents may have a coroutine ``llm.complete`` and
coroutine tools; plain sync ones are run in worker threads, so existing agents work
unchanged. Adds a per-task timeout and a ``stream()`` iterator that yields results as
they land. Cancelling the run cancels in-flight tasks.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

from .core import Agent, Orchestrator, Result, Task, _Safe


async def _maybe_await(fn, *args):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    out = await asyncio.to_thread(fn, *args)
    return await out if inspect.isawaitable(out) else out


async def arun_agent(agent: Agent, prompt: str) -> str:
    """Async twin of ``Agent.run`` (same protocol, same step limit)."""
    messages = [{"role": "system", "content": agent.system or f"You are {agent.name}. Tools: {agent.tools.names()}"},
                {"role": "user", "content": prompt}]
    for _ in range(agent.max_steps):
        raw = await _maybe_await(agent.llm.complete, messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                raise ValueError
        except ValueError:
            return raw
        if "final" in msg:
            return str(msg["final"])
        if "tool" in msg:
            try:
                obs = await _maybe_await(agent.tools.call, msg["tool"], msg.get("args", {}))
            except Exception as e:
                obs = f"ERROR: {e}"
            messages.append({"role": "user", "content": f"observation: {obs}"})
        else:
            return raw
    raise RuntimeError(f"{agent.name} exceeded {agent.max_steps} steps")


class AsyncOrchestrator:
    def __init__(self, agents: List[Agent], max_concurrency: int = 8,
                 backoff: float = 0.0, timeout: Optional[float] = None):
        self.agents = {a.name: a for a in agents}
        self.max_concurrency = max_concurrency
        self.backoff = backoff
        self.timeout = timeout  # seconds per attempt, None = unlimited
        self._planner = Orchestrator(agents)

    def plan(self, tasks: List[Task]) -> List[List[str]]:
        return self._planner.plan(tasks)

    async def _attempt(self, task: Task, prompt: str) -> str:
        coro = arun_agent(self.agents[task.agent], prompt)
        if self.timeout is None:
            return await coro
        return await asyncio.wait_for(coro, self.timeout)

    async def _exec(self, task: Task, outputs: Dict[str, str], sem: asyncio.Semaphore) -> Result:
        prompt = task.prompt.format_map(_Safe(outputs))
        t0 = time.perf_counter()
        err = None
        async with sem:
            for attempt in range(1, task.retries + 2):
                try:
                    out = await self._attempt(task, prompt)
                    return Result("ok", out, attempts=attempt, duration=time.perf_counter() - t0)
                except asyncio.TimeoutError:
                    err = f"timeout after {self.timeout}s"
                except Exception as e:
                    err = str(e)
                if attempt <= task.retries:
                    await asyncio.sleep(self.backoff * 2 ** (attempt - 1))
        return Result("failed", error=err, attempts=task.retries + 1, duration=time.perf_counter() - t0)

    async def stream(self, tasks: List[Task]) -> AsyncIterator[Tuple[str, Result]]:
        """Yield ``(task_id, Result)`` as each task finishes (skips included)."""
        self.plan(tasks)
        sem = asyncio.Semaphore(self.max_concurrency)
        pending = {t.id: t for t in tasks}
        results: Dict[str, Result] = {}
        running: Dict[asyncio.Task, str] = {}
        try:
            while pending or running:
                for tid, t in list(pending.items()):
                    if any(d in results and results[d].status != "ok" for d in t.deps):
                        results[tid] = Result("skipped", error="upstream failure")
                        del pending[tid]
                        yield tid, results[tid]
                    elif all(d in results for d in t.deps):
                        outs = {k: r.output for k, r in results.items() if r.output is not None}
                        running[asyncio.ensure_future(self._exec(t, outs, sem))] = tid
                        del pending[tid]
                if running:
                    done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                    for f in done:
                        tid = running.pop(f)
                        results[tid] = f.result()
                        yield tid, results[tid]
        finally:
            for f in running:
                f.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

    async def run(self, tasks: List[Task]) -> Dict[str, Result]:
        results: Dict[str, Result] = {}
        async for tid, r in self.stream(tasks):
            results[tid] = r
        return results

    def run_sync(self, tasks: List[Task]) -> Dict[str, Result]:
        return asyncio.run(self.run(tasks))
