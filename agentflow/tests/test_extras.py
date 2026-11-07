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
