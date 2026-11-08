import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *


class Slow:
    def __init__(self, d): self.d = d
    def complete(self, m):
        time.sleep(self.d); return "ok"


def ag(n, d): return Agent(n, Slow(d), ToolRegistry(), system=n)


def test_metrics_critical_path_and_parallelism():
    o = Orchestrator([ag("a", 0.05), ag("b", 0.15)])
    tasks = [Task("x", "a", "p"), Task("y", "b", "p"), Task("z", "a", "{x}{y}", deps=["x", "y"])]
    res = o.run(tasks)
    m = compute_metrics(tasks, res)
    assert m.critical_tasks == ["y", "z"]
    assert abs(m.critical_path - (res["y"].duration + res["z"].duration)) < 1e-9
    assert m.total_work > m.critical_path and m.parallelism > 1.0
    assert m.failed == m.skipped == m.retries == 0


def test_metrics_counts_failures_and_retries():
    class Boom:
        def complete(self, m): raise RuntimeError("x")
    o = Orchestrator([Agent("f", Boom(), ToolRegistry(), system="f"), ag("a", 0)])
    tasks = [Task("f1", "f", "p", retries=2), Task("d", "a", "{f1}", deps=["f1"])]
    m = compute_metrics(tasks, o.run(tasks))
    assert (m.failed, m.skipped, m.retries) == (1, 1, 2)
