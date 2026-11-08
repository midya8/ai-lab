"""Observed-latency tracking.

The catalogue's ``latency_ms`` / ``ms_per_token`` are static guesses. A
``LatencyTracker`` keeps a sliding window of the ratio *observed / predicted*
latency per model and exposes the window median as a correction factor, so the
``fastest`` and ``balanced`` strategies route around a model that is currently
slow (or prefer one that is currently quick). The median makes it robust to a
single outlier; ``min_samples`` keeps it neutral (factor 1.0) until it has
evidence.
"""

from __future__ import annotations

import statistics
from collections import deque


class LatencyTracker:
    def __init__(self, window: int = 20, min_samples: int = 3,
                 floor: float = 0.1, ceiling: float = 10.0) -> None:
        if window < 1 or min_samples < 1 or min_samples > window:
            raise ValueError("need 1 <= min_samples <= window")
        if not 0 < floor <= 1 <= ceiling:
            raise ValueError("need 0 < floor <= 1 <= ceiling")
        self.window, self.min_samples = window, min_samples
        self.floor, self.ceiling = floor, ceiling
        self._ratios: dict[str, deque] = {}

    def record(self, model: str, predicted_ms: float, observed_ms: float) -> None:
        if predicted_ms <= 0 or observed_ms < 0:
            return
        q = self._ratios.setdefault(model, deque(maxlen=self.window))
        q.append(observed_ms / predicted_ms)

    def samples(self, model: str) -> int:
        return len(self._ratios.get(model, ()))

    def factor(self, model: str) -> float:
        q = self._ratios.get(model)
        if not q or len(q) < self.min_samples:
            return 1.0
        return min(self.ceiling, max(self.floor, statistics.median(q)))

    def snapshot(self) -> dict:
        return {m: {"samples": len(q), "factor": round(self.factor(m), 3)}
                for m, q in sorted(self._ratios.items())}
