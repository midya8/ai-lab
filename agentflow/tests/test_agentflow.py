import os, sys, threading, time
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *


def mk(name, replies, tools=None):
    reg = ToolRegistry()
    for k, f in (tools or {}).items():
        reg.register(k, f)
    return Agent(name, ScriptedLLM(replies), reg)


def test_tool_loop_and_error_recovery():
    a = mk("a", [{"tool": "nope", "args": {}}, {"tool": "add", "args": {"x": 2, "y": 3}}, {"final": "5"}],
           {"add": lambda x, y: x + y})
    assert a.run("go") == "5"


def test_plain_text_and_step_limit():
    assert mk("a", ["hello"]).run("x") == "hello"
    loop = mk("b", [{"tool": "t", "args": {}}] * 10, {"t": lambda: 1})
    loop.max_steps = 3
    with pytest.raises(RuntimeError):
        loop.run("x")


def test_plan_layers_and_cycle():
    o = Orchestrator([mk("a", [])])
    layers = o.plan([Task("c", "a", "", ["a", "b"]), Task("a", "a", ""), Task("b", "a", "", ["a"])])
    assert layers == [["a"], ["b"], ["c"]]
    with pytest.raises(CycleError):
        o.plan([Task("x", "a", "", ["y"]), Task("y", "a", "", ["x"])])
    with pytest.raises(KeyError):
        o.plan([Task("x", "ghost", "")])


def test_dataflow_between_tasks():
    r = mk("r", ["facts"])
    w = mk("w", ["draft"])
    seen = []
    w.llm.complete = lambda m: (seen.append(m[-1]["content"]), "draft")[1]
    res = Orchestrator([r, w]).run([Task("research", "r", "find"), Task("write", "w", "use {research}", ["research"])])
    assert res["write"].output == "draft" and "use facts" in seen[0]


def test_parallel_execution():
    barrier = threading.Barrier(2, timeout=2)
    class L:
        def complete(self, m):
            barrier.wait()
            return "ok"
    ags = [Agent("p", L()), Agent("q", L())]
    res = Orchestrator(ags).run([Task("1", "p", "x"), Task("2", "q", "y")])
    assert all(r.status == "ok" for r in res.values())


def test_retry_then_success_and_failure_cascade():
    class Flaky:
        n = 0
        def complete(self, m):
            Flaky.n += 1
            if Flaky.n < 3:
                raise RuntimeError("boom")
            return "fine"
    o = Orchestrator([Agent("f", Flaky()), mk("g", [])])
    res = o.run([Task("t", "f", "x", retries=2)])
    assert res["t"].status == "ok" and res["t"].attempts == 3
    res = o.run([Task("bad", "g", "x"), Task("after", "g", "y", ["bad"])])
    assert res["bad"].status == "failed" and res["after"].status == "skipped"
