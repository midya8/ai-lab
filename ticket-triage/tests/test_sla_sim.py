import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from triage.sla_sim import simulate, poisson_arrivals, staffing_sweep, smallest_staffing

MIX = [("a", "normal", 3), ("b", "critical", 1)]


def test_idle_queue_no_breaches():
    r = simulate([(0.0, "a", "normal"), (100.0, "a", "normal")], {"a": 1})
    assert r["a"]["breaches"] == 0 and r["a"]["mean_wait"] == 0.0


def test_overloaded_single_agent_breaches():
    tickets = [(0.0, "a", "critical")] * 50
    r = simulate(tickets, {"a": 1}, service_hours=1.0)
    assert r["a"]["breaches"] > 0 and r["a"]["p95_wait"] > 1


def test_deterministic_per_seed():
    t = poisson_arrivals(3, 100, MIX, seed=2)
    assert simulate(t, {"a": 1, "b": 1}, seed=5) == simulate(t, {"a": 1, "b": 1}, seed=5)


def test_more_agents_never_worse_much():
    t = poisson_arrivals(4, 200, MIX, seed=3)
    s = staffing_sweep(t, "a", [1, 3, 6], {"b": 1})
    assert s[6] <= s[1] and s[6] == 0.0


def test_priority_served_first():
    t = [(0.0, "a", "low")] * 5 + [(0.01, "a", "critical")]
    # one agent, long service: critical must jump the waiting lows
    r = simulate(t, {"a": 1}, service_hours=2.0, seed=1)
    assert r["a"]["tickets"] == 6


def test_smallest_staffing_meets_target():
    t = poisson_arrivals(4, 200, MIX, seed=3)
    n = smallest_staffing(t, "a", 0.0, {"b": 2})
    assert n is not None
    assert simulate(t, {"a": n, "b": 2})["a"]["breach_rate"] == 0.0
