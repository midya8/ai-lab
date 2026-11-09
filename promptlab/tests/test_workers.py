import threading
import time

import pytest

from promptlab.cost import BudgetExceeded, CostMeter
from promptlab.registry import Registry
from promptlab.runner import Case, run_prompt
from promptlab.template import Template


class V:
    name, number, hash = "p", 1, "h"
    template = Template("say {{x}}")


def cases(n):
    return [Case(f"c{i}", {"x": str(i)}, [{"type": "contains", "value": str(i)}]) for i in range(n)]


def echo(prompt):
    return prompt


def test_parallel_matches_serial_and_keeps_order():
    cs = cases(20)
    a = run_prompt(V, cs, echo)
    b = run_prompt(V, cs, echo, workers=6)
    assert [r.id for r in b.results] == [c.id for c in cs]
    assert a.to_dict() == b.to_dict()


def test_workers_actually_overlap_and_speed_up():
    live, peak, lock = 0, 0, threading.Lock()

    def slow(prompt):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return prompt

    t = time.time()
    run_prompt(V, cases(12), slow, workers=6)
    assert peak > 1
    assert time.time() - t < 12 * 0.05 * 0.6


def test_model_error_isolated_per_case():
    def flaky(prompt):
        if "3" in prompt:
            raise RuntimeError("boom")
        return prompt

    r = run_prompt(V, cases(6), flaky, workers=3).by_id()
    assert not r["c3"].passed and "boom" in r["c3"].error
    assert r["c4"].passed


def test_budget_stops_parallel_run_and_meter_is_consistent():
    m = CostMeter(echo, 1_000_000, 0, budget=0.00001)  # roughly a couple of calls
    with pytest.raises(BudgetExceeded):
        run_prompt(V, cases(30), m, workers=5)
    assert m.usage.cost <= 0.00001 + 1e-9
    assert m.usage.calls == m.usage.tokens_in // max(1, m.count("say 0"))


def test_meter_counts_exactly_under_threads():
    m = CostMeter(echo, 1.0, 1.0)
    run_prompt(V, cases(50), m, samples=2, workers=8)
    assert m.usage.calls == 100


def test_bad_workers():
    with pytest.raises(ValueError):
        run_prompt(V, cases(1), echo, workers=0)
