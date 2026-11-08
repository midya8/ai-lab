"""Use a Supervisor as a node inside an Orchestrator DAG."""
from typing import Optional


class SupervisedAgent:
    """Adapter exposing a Supervisor through the Agent interface (name + run).

    Tasks addressed to this agent are routed to the best worker (with memory,
    fallbacks and circuit breaking) instead of a fixed one. The last routing
    decision per task prompt is kept in ``decisions`` for inspection.
    """

    def __init__(self, supervisor, name: str = "auto"):
        self.supervisor = supervisor
        self.name = name
        self.system = "routes to the best worker"
        self.decisions = []

    def run(self, prompt: str) -> str:
        res = self.supervisor.run(prompt)
        self.decisions.append({k: v for k, v in res.items() if k != "output"})
        return res["output"]

    def routing_summary(self) -> dict:
        counts = {}
        for d in self.decisions:
            counts[d["agent"]] = counts.get(d["agent"], 0) + 1
        return counts
