"""SLA-breach simulation (synthetic): discrete-event queue per routed queue.

Tickets arrive as a Poisson process, are routed by priority, and are served by
a fixed number of agents per queue (priority-first, FIFO within priority).
A ticket breaches when its time-to-first-response exceeds its SLA hours.
All numbers are synthetic and only show how staffing affects breach rates.
"""
import heapq
import random

from .routing import PRIORITIES, SLA_HOURS


def simulate(tickets, agents, service_hours=0.5, seed=0):
    """tickets: list of (arrival_hour, queue, priority), any order.
    agents: {queue: n_agents}. Returns {queue: stats}. Deterministic per seed."""
    rng = random.Random(seed)
    by_queue = {}
    for arrival, queue, pri in sorted(tickets):
        by_queue.setdefault(queue, []).append((arrival, pri))
    out = {}
    for queue, items in by_queue.items():
        n = max(1, agents.get(queue, 1))
        free = [0.0] * n  # time each agent becomes free
        heapq.heapify(free)
        waiting, waits, breaches, i = [], [], 0, 0
        total = len(items)
        done = 0
        clock = 0.0
        while done < total:
            t_free = free[0]
            # admit everything that arrived by the time the next agent is free
            if not waiting and i < total and items[i][0] > t_free:
                t_free = items[i][0]
            clock = t_free
            while i < total and items[i][0] <= clock:
                a, p = items[i]
                heapq.heappush(waiting, (-PRIORITIES.index(p), a, i, p))
                i += 1
            _, a, _, p = heapq.heappop(waiting)
            start = max(clock, a)
            heapq.heappop(free)
            heapq.heappush(free, start + rng.expovariate(1.0 / service_hours))
            wait = start - a
            waits.append(wait)
            breaches += wait > SLA_HOURS[p]
            done += 1
        waits.sort()
        out[queue] = {"tickets": total, "breaches": breaches,
                      "breach_rate": breaches / total,
                      "mean_wait": sum(waits) / total,
                      "p95_wait": waits[min(total - 1, int(0.95 * total))]}
    return out


def poisson_arrivals(rate_per_hour, hours, mix, seed=0):
    """mix: list of (queue, priority, weight). Returns tickets list."""
    rng = random.Random(seed)
    weights = [w for _, _, w in mix]
    t, tickets = 0.0, []
    while True:
        t += rng.expovariate(rate_per_hour)
        if t > hours:
            return tickets
        q, p, _ = rng.choices(mix, weights)[0]
        tickets.append((t, q, p))


def staffing_sweep(tickets, queue, counts, base_agents, **kw):
    """Breach rate of one queue as its agent count varies."""
    return {n: simulate(tickets, {**base_agents, queue: n}, **kw)[queue]["breach_rate"]
            for n in counts}


def smallest_staffing(tickets, queue, target, base_agents, max_agents=20, **kw):
    """Fewest agents keeping the queue's breach rate <= target, or None."""
    for n in range(1, max_agents + 1):
        if simulate(tickets, {**base_agents, queue: n}, **kw)[queue]["breach_rate"] <= target:
            return n
    return None


def _demo():
    mix = [("billing", "normal", 4), ("bug", "high", 2), ("security-response", "critical", 1),
           ("bug", "normal", 3)]
    tickets = poisson_arrivals(6.0, 24 * 14, mix, seed=1)
    base = {"billing": 2, "bug": 2, "security-response": 1}
    print("queue  agents -> breach rate (synthetic)")
    for q in ("bug", "security-response"):
        for n, r in staffing_sweep(tickets, q, [1, 2, 3, 4], base).items():
            print(f"{q:18s} {n} -> {r:.1%}")
        print(f"{q}: smallest staffing for <=1% breaches:",
              smallest_staffing(tickets, q, 0.01, base))


if __name__ == "__main__":
    _demo()
