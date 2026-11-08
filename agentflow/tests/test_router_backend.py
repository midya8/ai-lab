import os, sys, threading
from types import SimpleNamespace
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "llm-router"))
from agentflow import *
from agentflow.router_backend import render_messages


class FakeRouter:
    """Duck-typed stand-in: records calls, returns scripted replies."""

    def __init__(self, replies=None, cost=0.001, fail=False):
        self.replies = list(replies or [])
        self.cost, self.fail = cost, fail
        self.calls = []
        self.active = 0
        self.max_active = 0

    def route(self, prompt, max_tokens=256, strategy=None, verifier=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                               "strategy": strategy, "verifier": verifier})
            if self.fail:
                raise RuntimeError("all models failed")
            text = self.replies.pop(0) if self.replies else "ok"
            return SimpleNamespace(text=text, model="m-" + str(strategy), strategy=strategy or "balanced",
                                   cost=self.cost, latency_ms=10.0, cached=False, fallbacks=0, verified=None)
        finally:
            self.active -= 1


def test_render_messages_is_stable_and_role_tagged():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert render_messages(msgs) == "[system] S\n[user] U"
    assert render_messages(msgs) == render_messages(list(msgs))


def test_llm_passes_strategy_and_max_tokens():
    r = FakeRouter()
    b = RouterBackend(r)
    out = b.llm("a", strategy="cheapest", max_tokens=64).complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert r.calls[0]["strategy"] == "cheapest" and r.calls[0]["max_tokens"] == 64
    assert r.calls[0]["prompt"] == "[user] hi"


def test_agent_runs_tool_loop_over_router():
    r = FakeRouter(['{"tool": "echo", "args": {"x": "ping"}}', '{"final": "got ping"}'])
    b = RouterBackend(r)
    tools = ToolRegistry(); tools.register("echo", lambda x: x)
    agent = Agent("a", b.llm("a"), tools)
    assert agent.run("go") == "got ping"
    assert len(b.records) == 2
    # second call carries the observation back to the model
    assert "observation: ping" in r.calls[1]["prompt"]


def test_report_aggregates_per_agent_and_model():
    r = FakeRouter(cost=0.002)
    b = RouterBackend(r)
    for name, strat, n in [("planner", "quality", 1), ("worker", "cheapest", 3)]:
        llm = b.llm(name, strategy=strat)
        for _ in range(n):
            llm.complete([{"role": "user", "content": "x"}])
    rep = b.report()
    assert rep["total"]["calls"] == 4
    assert rep["total"]["cost"] == pytest.approx(0.008)
    assert rep["per_agent"]["worker"]["calls"] == 3
    assert set(rep["per_model"]) == {"m-quality", "m-cheapest"}
    text = b.format_report()
    assert "planner" in text and "TOTAL" in text


def test_budget_cap_refuses_after_spend_reached():
    b = RouterBackend(FakeRouter(cost=0.01), max_cost=0.02)
    llm = b.llm("a")
    llm.complete([{"role": "user", "content": "1"}])
    llm.complete([{"role": "user", "content": "2"}])
    with pytest.raises(BudgetExhausted):
        llm.complete([{"role": "user", "content": "3"}])
    assert len(b.records) == 2


def test_router_failure_propagates_and_orchestrator_skips_downstream():
    b = RouterBackend(FakeRouter(fail=True))
    a = Agent("a", b.llm("a")); c = Agent("c", b.llm("c"))
    res = Orchestrator([a, c]).run([Task("t1", "a", "x"), Task("t2", "c", "{t1}", ["t1"])])
    assert res["t1"].status == "failed"
    assert res["t2"].status == "skipped"


def test_parallel_agents_do_not_overlap_router_calls():
    class Slow(FakeRouter):
        def route(self, *a, **k):
            import time; time.sleep(0.01)
            return super().route(*a, **k)
    r = Slow()
    b = RouterBackend(r)
    agents = [Agent(f"a{i}", b.llm(f"a{i}")) for i in range(4)]
    res = Orchestrator(agents, max_workers=4).run([Task(f"t{i}", f"a{i}", "go") for i in range(4)])
    assert all(x.status == "ok" for x in res.values())
    assert r.max_active == 1
    assert len(b.records) == 4


def test_json_or_text_verifier():
    assert json_or_text("plain answer")
    assert json_or_text('{"final": "x"}')
    assert json_or_text('{"tool": "t", "args": {}}')
    assert not json_or_text('{"oops": 1}')
    assert json_or_text("[1, 2]")


def test_verifier_is_forwarded():
    r = FakeRouter()
    b = RouterBackend(r)
    b.llm("a", verifier=json_or_text).complete([{"role": "user", "content": "q"}])
    assert r.calls[0]["verifier"] is json_or_text


def test_real_router_integration_strategies_pick_different_models():
    llmrouter = pytest.importorskip("llmrouter")
    models = llmrouter.load_models(os.path.join(HERE, "..", "..", "llm-router", "examples", "models.json"))
    router = llmrouter.Router(models, llmrouter.SimProvider(), cache=llmrouter.ResponseCache())
    b = RouterBackend(router)
    cheap = Agent("cheap", b.llm("cheap", strategy="cheapest"))
    best = Agent("best", b.llm("best", strategy="quality"))
    res = Orchestrator([cheap, best]).run([Task("a", "cheap", "Summarise: cats"),
                                           Task("b", "best", "Summarise: cats")])
    assert {r.status for r in res.values()} == {"ok"}
    models_used = {r["model"] for r in b.records}
    assert len(models_used) == 2
    cost = {r["agent"]: r["cost"] for r in b.records}
    assert cost["cheap"] < cost["best"]
    # identical request again hits the router cache for free
    again = b.llm("cheap", strategy="cheapest")
    again.complete([{"role": "system", "content": "You are cheap. Tools: []"}, {"role": "user", "content": "Summarise: cats"}])
    assert router.stats()["requests"] == 3
