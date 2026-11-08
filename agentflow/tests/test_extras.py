import os, sys, json
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *


def mk(name, replies, tools=None, system=""):
    reg = ToolRegistry()
    for k, f in (tools or {}).items():
        reg.register(k, f)
    return Agent(name, ScriptedLLM(replies), reg, system=system)


def test_tracer_records_llm_and_tool_spans():
    tr = Tracer()
    a = trace_agent(mk("a", [{"tool": "add", "args": {"x": 1, "y": 2}}, {"final": "3"}], {"add": lambda x, y: x + y}), tr)
    assert a.run("go") == "3"
    kinds = [(s.kind, s.name) for s in tr.spans]
    assert kinds == [("llm", "complete"), ("tool", "add"), ("llm", "complete")]
    assert tr.summary()["a"]["llm_calls"] == 2 and tr.summary()["a"]["tool_calls"] == 1
    assert len(json.loads(tr.to_json())) == 3


def test_tracer_marks_tool_errors_and_works_in_orchestrator():
    tr = Tracer()
    a = trace_agent(mk("a", [{"tool": "boom", "args": {}}, {"final": "ok"}], {"boom": lambda: 1 / 0}), tr)
    res = Orchestrator([a]).run([Task("t", "a", "x")])
    assert res["t"].status == "ok"
    assert tr.summary()["a"]["errors"] == 1


def test_supervisor_routes_and_falls_back():
    m = mk("math", ["42"], system="arithmetic")
    w = mk("writer", ["prose"], system="writing")
    s = Supervisor(ScriptedLLM([{"route": "math"}, {"route": "ghost"}, "writer"]), [w, m], default="writer")
    assert s.run("2*21") == {"agent": "math", "output": "42"}
    assert s.route("x") == "writer"  # invalid -> default
    assert s.route("y") == "writer"  # plain-text route


def test_supervisor_validation():
    with pytest.raises(ValueError):
        Supervisor(ScriptedLLM([]), [])
    with pytest.raises(KeyError):
        Supervisor(ScriptedLLM([]), [mk("a", [])], default="zz")


def test_supervisor_memory_injects_context_and_records_result():
    from agentflow import SharedMemory
    seen = []

    class Echo:
        def complete(self, messages):
            seen.append(messages[-1]["content"])
            return json.dumps({"final": "ok:" + str(len(seen))})
    m = SharedMemory()
    m.remember("pricing", "enterprise tier costs more")
    w = Agent("a", Echo(), ToolRegistry())
    s = Supervisor(ScriptedLLM([{"route": "a"}, {"route": "a"}]), [w], memory=m)
    r = s.run("what about pricing?")
    assert r["memory_used"] == ["pricing"]
    assert "enterprise tier costs more" in seen[0]
    assert m.recall("supervisor:what about pricing?") == r["output"]
    r2 = s.run("pricing again")
    assert "supervisor:what about pricing?" in r2["memory_used"]


def test_supervisor_memory_optional_and_no_write_flag():
    from agentflow import SharedMemory
    m = SharedMemory()
    w = mk("a", [{"final": "x"}])
    s = Supervisor(ScriptedLLM([{"route": "a"}]), [w], memory=m, remember_results=False)
    assert s.run("hi")["memory_used"] == [] and m.keys() == []
    plain = Supervisor(ScriptedLLM([{"route": "a"}]), [mk("a", [{"final": "x"}])])
    assert "memory_used" not in plain.run("hi")
