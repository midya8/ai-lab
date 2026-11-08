import os, sys, json, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *


def mk(name, replies):
    return Agent(name, ScriptedLLM(replies), ToolRegistry())


def test_remember_recall_forget():
    m = SharedMemory()
    m.remember("k", "v", "a")
    assert m.recall("k") == "v" and m.recall("x") is None
    assert m.forget("k") and not m.forget("k")


def test_search_ranks_key_matches_higher():
    m = SharedMemory()
    m.remember("pricing", "enterprise tier costs more")
    m.remember("notes", "pricing was discussed briefly")
    m.remember("other", "unrelated")
    r = m.search("pricing")
    assert [e["key"] for e in r] == ["pricing", "notes"]


def test_agents_share_via_tools_in_orchestrator():
    m = SharedMemory()
    w = attach_memory(mk("writer", [{"tool": "remember", "args": {"key": "fact", "value": "sky is blue"}}, {"final": "done"}]), m)
    r = attach_memory(mk("reader", [{"tool": "recall", "args": {"key": "fact"}}, {"final": "got it"}]), m)
    res = Orchestrator([w, r]).run([Task("a", "writer", "store"), Task("b", "reader", "read {a}")])
    assert res["a"].status == "ok" and res["b"].status == "ok"
    assert m.recall("fact") == "sky is blue"
    assert m._data["fact"]["author"] == "writer"


def test_persistence_roundtrip(tmp_path):
    p = str(tmp_path / "mem.json")
    SharedMemory(p).remember("a", "b", "x")
    assert SharedMemory(p).recall("a") == "b"
    assert json.load(open(p))["a"]["author"] == "x"
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_thread_safety():
    m = SharedMemory()
    ts = [threading.Thread(target=lambda i=i: m.remember(f"k{i}", "v")) for i in range(50)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(m.keys()) == 50
