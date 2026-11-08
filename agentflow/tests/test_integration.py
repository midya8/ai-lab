import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *


class Boom:
    def complete(self, m):
        raise RuntimeError("down")


def make(router_replies, max_attempts=1):
    coder = Agent("coder", ScriptedLLM(["code!"] * 5), system="writes code")
    writer = Agent("writer", ScriptedLLM(["prose!"] * 5), system="writes prose")
    sup = Supervisor(ScriptedLLM(router_replies), [coder, writer], max_attempts=max_attempts)
    return coder, writer, SupervisedAgent(sup)


def test_dag_routes_each_task():
    coder, writer, auto = make(['{"route": "coder"}', '{"route": "writer"}'])
    orch = Orchestrator([auto])
    res = orch.run([Task("a", "auto", "write function"), Task("b", "auto", "summarise {a}", ["a"])])
    assert res["a"].output == "code!" and res["b"].output == "prose!"
    assert auto.routing_summary() == {"coder": 1, "writer": 1}


def test_fallback_visible_in_decisions():
    bad = Agent("bad", Boom(), system="x")
    good = Agent("good", ScriptedLLM(["ok"]), system="y")
    auto = SupervisedAgent(Supervisor(ScriptedLLM(["bad"]), [bad, good], max_attempts=2))
    res = Orchestrator([auto]).run([Task("t", "auto", "go")])
    assert res["t"].status == "ok" and res["t"].output == "ok"
    assert auto.decisions[0]["failed_attempts"] == ["bad"]


def test_total_failure_marks_task_failed_and_skips_downstream():
    bad = Agent("bad", Boom())
    auto = SupervisedAgent(Supervisor(ScriptedLLM(["bad"]), [bad]))
    res = Orchestrator([auto]).run([Task("t", "auto", "go"), Task("u", "auto", "{t}", ["t"])])
    assert res["t"].status == "failed" and res["u"].status == "skipped"


def test_mixed_fixed_and_supervised_agents():
    coder, writer, auto = make(['{"route": "writer"}'])
    res = Orchestrator([coder, auto]).run([Task("x", "coder", "q"), Task("y", "auto", "polish {x}", ["x"])])
    assert [res[k].status for k in "xy"] == ["ok", "ok"]
