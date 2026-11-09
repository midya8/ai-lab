import asyncio
import json
import threading
import time

import pytest

from agentflow import Agent, AsyncOrchestrator, Orchestrator, ScriptedLLM, Task, ToolRegistry, CycleError


class SlowLLM:
    """Coroutine LLM that records peak concurrency."""
    active = 0
    peak = 0

    def __init__(self, delay=0.05, reply="done"):
        self.delay, self.reply = delay, reply

    async def complete(self, messages):
        SlowLLM.active += 1
        SlowLLM.peak = max(SlowLLM.peak, SlowLLM.active)
        await asyncio.sleep(self.delay)
        SlowLLM.active -= 1
        return json.dumps({"final": self.reply})


def setup_function(_):
    SlowLLM.active = SlowLLM.peak = 0


def test_parallel_beats_serial_and_matches_sync_results():
    ag = Agent("a", SlowLLM(0.1, "x"))
    tasks = [Task(f"t{i}", "a", "p") for i in range(10)]
    t0 = time.perf_counter()
    res = AsyncOrchestrator([ag], max_concurrency=10).run_sync(tasks)
    assert time.perf_counter() - t0 < 0.5  # serial would be 1.0s
    assert all(r.status == "ok" and r.output == "x" for r in res.values())


def test_semaphore_bounds_concurrency():
    ag = Agent("a", SlowLLM(0.02))
    AsyncOrchestrator([ag], max_concurrency=3).run_sync([Task(f"t{i}", "a", "p") for i in range(12)])
    assert SlowLLM.peak == 3


def test_dag_dependencies_and_templating():
    a = Agent("a", ScriptedLLM([json.dumps({"final": "alpha"}), json.dumps({"final": "got alpha"})]))
    tasks = [Task("one", "a", "go"), Task("two", "a", "use {one}", deps=["one"])]
    res = AsyncOrchestrator([a]).run_sync(tasks)
    assert res["two"].status == "ok" and res["two"].output == "got alpha"


def test_sync_llm_and_sync_tools_run_in_threads():
    seen = []
    tools = ToolRegistry()
    tools.register("who", lambda: seen.append(threading.current_thread().name) or "bob")
    llm = ScriptedLLM([json.dumps({"tool": "who", "args": {}}), json.dumps({"final": "hi bob"})])
    res = AsyncOrchestrator([Agent("a", llm, tools)]).run_sync([Task("t", "a", "p")])
    assert res["t"].output == "hi bob"
    assert seen and seen[0] != threading.current_thread().name


def test_async_tools():
    class L:
        def __init__(self):
            self.n = 0

        async def complete(self, m):
            self.n += 1
            return json.dumps({"tool": "t", "args": {}} if self.n == 1 else {"final": m[-1]["content"]})

    tools = ToolRegistry()

    async def t():
        await asyncio.sleep(0)
        return "async-result"
    tools.register("t", t)
    res = AsyncOrchestrator([Agent("a", L(), tools)]).run_sync([Task("t", "a", "p")])
    assert res["t"].output == "observation: async-result"


def test_failure_cascades_and_retries():
    class Flaky:
        def __init__(self):
            self.n = 0

        async def complete(self, m):
            self.n += 1
            if self.n < 3:
                raise RuntimeError("boom")
            return json.dumps({"final": "ok"})

    class Dead:
        async def complete(self, m):
            raise RuntimeError("dead")

    flaky, dead = Flaky(), Dead()
    tasks = [Task("f", "flaky", "p", retries=2), Task("d", "dead", "p"),
             Task("after", "flaky", "{d}", deps=["d"]), Task("sib", "flaky", "{f}", deps=["f"])]
    res = AsyncOrchestrator([Agent("flaky", flaky), Agent("dead", dead)]).run_sync(tasks)
    assert res["f"].status == "ok" and res["f"].attempts == 3
    assert res["d"].status == "failed" and res["d"].error == "dead"
    assert res["after"].status == "skipped"
    assert res["sib"].status == "ok"


def test_timeout_is_a_failure_and_retry_can_recover():
    class SometimesSlow:
        n = 0

        async def complete(self, m):
            SometimesSlow.n += 1
            await asyncio.sleep(1.0 if SometimesSlow.n == 1 else 0)
            return json.dumps({"final": "fast"})

    SometimesSlow.n = 0
    orch = AsyncOrchestrator([Agent("a", SometimesSlow())], timeout=0.05)
    res = orch.run_sync([Task("t", "a", "p", retries=1)])
    assert res["t"].status == "ok" and res["t"].attempts == 2
    SometimesSlow.n = 0
    res = orch.run_sync([Task("t", "a", "p")])
    assert res["t"].status == "failed" and "timeout" in res["t"].error


def test_stream_yields_in_completion_order():
    class ByPrompt:
        async def complete(self, m):
            d = float(m[1]["content"])
            await asyncio.sleep(d)
            return json.dumps({"final": str(d)})

    tasks = [Task("slow", "a", "0.15"), Task("fast", "a", "0.01"), Task("mid", "a", "0.06")]

    async def go():
        return [tid async for tid, _ in AsyncOrchestrator([Agent("a", ByPrompt())]).stream(tasks)]
    assert asyncio.run(go()) == ["fast", "mid", "slow"]


def test_early_exit_cancels_inflight():
    class Hang:
        cancelled = False

        async def complete(self, m):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                Hang.cancelled = True
                raise

    async def go():
        orch = AsyncOrchestrator([Agent("a", Hang())])
        agen = orch.stream([Task("t", "a", "p")])
        nxt = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)
        nxt.cancel()
        await asyncio.gather(nxt, return_exceptions=True)
        await agen.aclose()
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert Hang.cancelled


def test_bad_graphs_raise_like_sync():
    a = Agent("a", ScriptedLLM([]))
    with pytest.raises(CycleError):
        AsyncOrchestrator([a]).run_sync([Task("x", "a", "p", deps=["y"]), Task("y", "a", "p", deps=["x"])])
    with pytest.raises(KeyError):
        AsyncOrchestrator([a]).run_sync([Task("x", "nope", "p")])


def test_matches_sync_orchestrator_on_same_dag():
    def mk():
        return Agent("a", ScriptedLLM([json.dumps({"final": "v"})] * 5))
    tasks = [Task("a1", "a", "p"), Task("a2", "a", "p"), Task("c", "a", "{a1}{a2}", deps=["a1", "a2"])]
    s = Orchestrator([mk()], max_workers=1).run(tasks)
    a = AsyncOrchestrator([mk()], max_concurrency=1).run_sync(tasks)
    assert {k: (r.status, r.output) for k, r in s.items()} == {k: (r.status, r.output) for k, r in a.items()}
