"""Deterministic synthetic log generator with injected incidents."""
import random
from datetime import datetime, timedelta

NORMAL = [
    ("INFO", "GET /api/users/{n} 200 {ms}ms from 10.0.{a}.{b}"),
    ("INFO", "POST /api/orders 201 {ms}ms from 10.0.{a}.{b}"),
    ("INFO", "cache hit key={n}"),
    ("WARN", "slow query took {ms}ms"),
]


def generate(minutes=30, seed=0, incidents=True):
    rng = random.Random(seed)
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    lines = []
    for m in range(minutes):
        n = rng.randint(55, 65)
        spike = incidents and m == 20
        burst = incidents and m == 25
        if spike:
            n *= 8
        for _ in range(n):
            lvl, tpl = rng.choice(NORMAL)
            msg = tpl.format(n=rng.randint(1, 9999), ms=rng.randint(3, 400), a=rng.randint(0, 9), b=rng.randint(1, 250))
            ts = t0 + timedelta(minutes=m, seconds=rng.random() * 59)
            lines.append((ts, lvl, msg))
        if burst:
            for _ in range(30):
                ts = t0 + timedelta(minutes=m, seconds=rng.random() * 59)
                lines.append((ts, "ERROR", f"db connection refused host=10.0.0.{rng.randint(1,9)} retry={rng.randint(1,5)}"))
    lines.sort()
    return [f"{ts.isoformat(timespec='seconds')} {lvl} {msg}" for ts, lvl, msg in lines]
