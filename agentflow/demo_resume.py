"""Crash-and-resume: a flaky tool fails a pipeline once; the rerun reuses finished work."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agentflow import *


class Counting:
    def __init__(self, name, fail_first=False):
        self.name, self.calls, self.fail_first = name, 0, fail_first

    def complete(self, messages):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("upstream API timeout")
        return f"[{self.name} output]"


def pipeline(flaky):
    llms = {n: Counting(n, fail_first=(n == "writer" and flaky)) for n in ("scraper_a", "scraper_b", "writer")}
    return Orchestrator([Agent(n, l, ToolRegistry(), system=n) for n, l in llms.items()]), llms


tasks = [Task("a", "scraper_a", "scrape A"), Task("b", "scraper_b", "scrape B"),
         Task("report", "writer", "write from {a} and {b}", deps=["a", "b"])]
log = RunLog(os.path.join(tempfile.mkdtemp(), "run.jsonl"))

o, llms = pipeline(flaky=True)
r = o.run(tasks, log=log)
print("run 1:", {k: v.status for k, v in r.items()}, "| LLM calls:", {n: l.calls for n, l in llms.items()})
o, llms = pipeline(flaky=False)
r = o.run(tasks, log=log, resume=True)
print("run 2:", {k: v.status + ("*" if v.resumed else "") for k, v in r.items()}, "| LLM calls:", {n: l.calls for n, l in llms.items()})
print("(* = replayed from the log)\n")
print(log.format_runs())
