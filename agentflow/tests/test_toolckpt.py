import json

from agentflow import Agent, AsyncOrchestrator, ScriptedLLM, Task, ToolRegistry, ToolJournal, RunLog


class FlakyLLM:
    """Calls send, then charge, then fails on the first run before finishing."""
    def __init__(self, fail_first=True):
        self.fail = fail_first
        self.i = 0

    async def complete(self, messages):
        self.i += 1
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


def test_retry_does_not_repeat_side_effects(tmp_path):
    calls = []
    j = ToolJournal(str(tmp_path / "t.jsonl"))
    ag = make(calls)
    orch = AsyncOrchestrator([ag], tool_journal=j)
    r = orch.run_sync([Task("t", "a", "go", retries=1)])["t"]
    assert r.status == "ok" and r.attempts == 2
    assert calls == ["send", "charge"]  # each ran once despite the retry
    assert orch.tool_stats == {"replayed": 2, "live": 2}


def test_without_journal_side_effects_repeat():
    calls = []
    r = AsyncOrchestrator([make(calls)]).run_sync([Task("t", "a", "go", retries=1)])["t"]
    assert r.status == "ok" and calls == ["send", "charge", "send", "charge"]


def test_journal_survives_restart_and_resume_across_processes(tmp_path):
    path = str(tmp_path / "t.jsonl")
    calls = []
    AsyncOrchestrator([make(calls)], tool_journal=ToolJournal(path)).run_sync([Task("t", "a", "go")])
    assert calls == ["send", "charge"]  # task failed after two calls
    assert len(ToolJournal(path)) == 2
    # new "process": fresh journal object reads the file
    orch = AsyncOrchestrator([make(calls, fail_first=False)], tool_journal=ToolJournal(path))
    r = orch.run_sync([Task("t", "a", "go")])["t"]
    assert r.status == "ok" and calls == ["send", "charge"]
    assert orch.tool_stats["replayed"] == 2


def test_edited_prompt_invalidates_entries(tmp_path):
    path = str(tmp_path / "t.jsonl")
    calls = []
    AsyncOrchestrator([make(calls)], tool_journal=ToolJournal(path)).run_sync([Task("t", "a", "go")])
    AsyncOrchestrator([make(calls, False)], tool_journal=ToolJournal(path)).run_sync([Task("t", "a", "go v2")])
    assert calls == ["send", "charge", "send", "charge"]


def test_divergence_stops_replay(tmp_path):
    j = ToolJournal(str(tmp_path / "t.jsonl"))
    j.store("fp", 0, "send", {"to": "a"}, "sent")
    j.store("fp", 1, "charge", {"amt": 5}, "charged 5")
    sc = j.scope("fp")
    ran = []
    fn = lambda t, a: ran.append((t, a)) or "live"
    assert sc.call_sync("send", {"to": "a"}, fn) == "sent"
    assert sc.call_sync("charge", {"amt": 9}, fn) == "live"  # args differ -> live
    assert sc.call_sync("x", {}, fn) == "live"
    assert len(ran) == 2 and sc.replayed == 1 and sc.live == 2


def test_failed_tool_calls_are_not_journaled(tmp_path):
    j = ToolJournal(str(tmp_path / "t.jsonl"))
    sc = j.scope("fp")
    def boom(t, a):
        raise ValueError("x")
    try:
        sc.call_sync("send", {}, boom)
    except ValueError:
        pass
    assert len(j) == 0


def test_corrupt_trailing_line_ignored(tmp_path):
    p = tmp_path / "t.jsonl"
    j = ToolJournal(str(p))
    j.store("fp", 0, "send", {}, "ok")
    with open(p, "a") as f:
        f.write('{"fp": "fp", "se')
    assert len(ToolJournal(str(p))) == 1


def test_composes_with_runlog_resume(tmp_path):
    calls = []
    log = RunLog(str(tmp_path / "r.jsonl"))
    j = ToolJournal(str(tmp_path / "t.jsonl"))
    tasks = [Task("t", "a", "go")]
    AsyncOrchestrator([make(calls)], tool_journal=j).run_sync(tasks, log=log)
    orch = AsyncOrchestrator([make(calls, False)], tool_journal=j)
    r = orch.run_sync(tasks, log=log, resume=True)["t"]
    assert r.status == "ok" and calls == ["send", "charge"]
