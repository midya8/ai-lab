"""Token-bucket rate limiter, one per model, so the router skips a model that
has exhausted its requests-per-minute quota instead of hitting a provider 429."""

from __future__ import annotations

from .clock import SystemClock


class TokenBucket:
    """Classic token bucket: ``rate_per_s`` tokens refill up to ``capacity``."""

    def __init__(self, rate_per_s: float, capacity: float | None = None, clock=None) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be > 0")
        self.rate = float(rate_per_s)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_s))
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._clock = clock or SystemClock()
        self._tokens = self.capacity
        self._last = self._clock.now()

    @classmethod
    def per_minute(cls, rpm: float, clock=None) -> "TokenBucket":
        return cls(rpm / 60.0, capacity=max(1.0, rpm), clock=clock)

    def _refill(self) -> None:
        now = self._clock.now()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def available(self, cost: float = 1.0) -> bool:
        """Peek without consuming."""
        return self.tokens >= cost - 1e-12

    def try_acquire(self, cost: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= cost - 1e-12:
            self._tokens -= cost
            return True
        return False

    def wait_time(self, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens will be available (0 if now)."""
        self._refill()
        missing = cost - self._tokens
        return max(0.0, missing / self.rate)
