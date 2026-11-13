import json

from agentflow import Agent, Orchestrator, Task, ToolJournal, ToolRegistry


class FlakyLLM:
    def __init__(self, fail_first=True):
        self.fail = fail_first

    def complete(self, messages):
        n = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("observation"))
        if n == 0:
            return json.dumps({"tool": "send", "args": {"to": "a"}})
        if n == 1:
            return json.dumps({"tool": "charge", "args": {"amt": 5}})
        if n == 2 and self.fail:
            self.fail = False
            raise RuntimeError("llm blew up")
        return json.dumps({"final": "done"})


def make(calls, fail_first=True):
    tools = ToolRegistry()
    tools.register("send", lambda to: calls.append("send") or "sent")
    tools.register("charge", lambda amt: calls.append("charge") or f"charged {amt}")
    return Agent("a", FlakyLLM(fail_first), tools)


def test_sync_retry_does_not_repeat_side_effects(tmp_path):
    calls = []
    orch = Orchestrator([make(calls)], tool_journal=ToolJournal(str(tmp_path / "t.jsonl")))
    r = orch.run([Task("t", "a", "go", retries=1)])["t"]
    assert r.status == "ok" and r.attempts == 2
    assert calls == ["send", "charge"]
    assert orch.tool_stats == {"replayed": 2, "live": 2}


def test_sync_without_journal_repeats():
    calls = []
    r = Orchestrator([make(calls)]).run([Task("t", "a", "go", retries=1)])["t"]
    assert r.status == "ok" and calls == ["send", "charge", "send", "charge"]


def test_sync_journal_shared_with_async_across_restart(tmp_path):
    from agentflow import AsyncOrchestrator
    path = str(tmp_path / "t.jsonl")
    calls = []
    Orchestrator([make(calls)], tool_journal=ToolJournal(path)).run([Task("t", "a", "go")])
    assert calls == ["send", "charge"]

    class AsyncOK:
        async def complete(self, messages):
            return FlakyLLM(False).complete(messages)
    tools = make(calls).tools
    orch = AsyncOrchestrator([Agent("a", AsyncOK(), tools)], tool_journal=ToolJournal(path))
    r = orch.run_sync([Task("t", "a", "go")])["t"]
    assert r.status == "ok" and calls == ["send", "charge"]
    assert orch.tool_stats["replayed"] == 2
