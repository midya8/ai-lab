"""Request metrics for the HTTP service: counters, latency percentiles, and
Prometheus text exposition. Stdlib only, thread-safe.

Latency percentiles use the nearest-rank method over a bounded window of the
most recent observations (default 1024), so memory stays constant. The
Prometheus histogram uses cumulative buckets over *all* observations.
"""
from __future__ import annotations

import math
import threading
from collections import Counter, deque

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def percentile(values, q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]. Empty input -> 0.0."""
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    xs = sorted(values)
    if not xs:
        return 0.0
    rank = max(1, math.ceil(q / 100 * len(xs)))
    return xs[rank - 1]


class Metrics:
    def __init__(self, window: int = 1024):
        self._lock = threading.Lock()
        self._lat = deque(maxlen=window)
        self._bucket_counts = [0] * len(BUCKETS)
        self._sum = 0.0
        self._count = 0
        self.requests: Counter = Counter()   # (route, status) -> n
        self.outcomes: Counter = Counter()   # answered / abstained
        self.cache: Counter = Counter()      # hit / miss

    def observe(self, route: str, status: int, seconds: float,
                outcome: str | None = None, cache: str | None = None) -> None:
        with self._lock:
            self.requests[(route, status)] += 1
            if outcome:
                self.outcomes[outcome] += 1
            if cache:
                self.cache[cache] += 1
            if route == "/ask" and status == 200:
                self._lat.append(seconds)
                self._sum += seconds
                self._count += 1
                for i, b in enumerate(BUCKETS):
                    if seconds <= b:
                        self._bucket_counts[i] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._lat)
            asked = self.outcomes["answered"] + self.outcomes["abstained"]
            return {
                "requests": {f"{r} {s}": n for (r, s), n in sorted(self.requests.items())},
                "answered": self.outcomes["answered"],
                "abstained": self.outcomes["abstained"],
                "abstain_rate": round(self.outcomes["abstained"] / asked, 4) if asked else 0.0,
                "cache_hits": self.cache["hit"], "cache_misses": self.cache["miss"],
                "latency_ms": {
                    "count": self._count,
                    "p50": round(percentile(lat, 50) * 1000, 3),
                    "p95": round(percentile(lat, 95) * 1000, 3),
                    "max": round(max(lat, default=0.0) * 1000, 3),
                },
            }

    def prometheus(self, cache_stats: dict | None = None) -> str:
        with self._lock:
            out = ["# HELP docqa_requests_total HTTP requests by route and status.",
                   "# TYPE docqa_requests_total counter"]
            for (r, s), n in sorted(self.requests.items()):
                out.append(f'docqa_requests_total{{route="{r}",status="{s}"}} {n}')
            out += ["# HELP docqa_answers_total Questions answered vs abstained.",
                    "# TYPE docqa_answers_total counter"]
            for k in ("answered", "abstained"):
                out.append(f'docqa_answers_total{{outcome="{k}"}} {self.outcomes[k]}')
            out += ["# HELP docqa_cache_requests_total Answer cache lookups.",
                    "# TYPE docqa_cache_requests_total counter"]
            for k in ("hit", "miss"):
                out.append(f'docqa_cache_requests_total{{result="{k}"}} {self.cache[k]}')
            out += ["# HELP docqa_ask_seconds Latency of successful /ask requests.",
                    "# TYPE docqa_ask_seconds histogram"]
            for b, n in zip(BUCKETS, self._bucket_counts):
                out.append(f'docqa_ask_seconds_bucket{{le="{b}"}} {n}')
            out.append(f'docqa_ask_seconds_bucket{{le="+Inf"}} {self._count}')
            out.append(f"docqa_ask_seconds_sum {self._sum:.6f}")
            out.append(f"docqa_ask_seconds_count {self._count}")
        if cache_stats is not None:
            out += ["# HELP docqa_cache_entries Entries currently cached.",
                    "# TYPE docqa_cache_entries gauge",
                    f"docqa_cache_entries {cache_stats['size']}"]
        return "\n".join(out) + "\n"
