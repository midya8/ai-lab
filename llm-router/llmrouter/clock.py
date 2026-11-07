"""Injectable clocks so cache TTLs, breaker cooldowns and budget windows
are testable without sleeping."""

from __future__ import annotations

import time


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    def advance(self, seconds: float) -> None:  # real time passes by itself
        return None


class ManualClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot move a clock backwards")
        self._t += seconds
