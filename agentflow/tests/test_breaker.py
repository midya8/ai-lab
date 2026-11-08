import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *


class Boom:
    def complete(self, m):
        raise RuntimeError("down")


def worker(name, llm):
    return Agent(name, llm, ToolRegistry(), system=name)


def test_breaker_state_machine():
    t = [0.0]
    b = CircuitBreaker(2, 10, clock=lambda: t[0])
    b.failure("a"); assert b.state("a") == "closed"
    b.failure("a"); assert b.state("a") == "open" and not b.allow("a")
    t[0] = 11; assert b.state("a") == "half-open" and b.allow("a")
    b.failure("a"); assert b.state("a") == "open"
    t[0] = 22; b.success("a"); assert b.state("a") == "closed"


def test_supervisor_skips_open_worker():
    br = CircuitBreaker(1, 100)
    bad, good = worker("bad", Boom()), worker("good", ScriptedLLM(["ok", "ok"]))
    s = Supervisor(ScriptedLLM(["bad", "bad"]), [bad, good], max_attempts=2, breaker=br)
    r1 = s.run("x")
    assert r1["agent"] == "good" and r1["failed_attempts"] == ["bad"]
    r2 = s.run("y")
    assert r2["agent"] == "good" and r2["skipped_open"] == ["bad"] and "failed_attempts" not in r2


def test_all_open_still_tries():
    br = CircuitBreaker(1, 100)
    br.failure("a")
    s = Supervisor(ScriptedLLM(["a"]), [worker("a", ScriptedLLM(["fine"]))], breaker=br)
    r = s.run("x")
    assert r["output"] == "fine" and "skipped_open" not in r
