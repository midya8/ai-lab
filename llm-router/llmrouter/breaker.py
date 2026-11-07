"""Per-model circuit breaker: closed -> open -> half-open -> closed."""

from __future__ import annotations

from .clock import SystemClock

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half-open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 clock=None) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock or SystemClock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return CLOSED
        if self._clock.now() - self._opened_at >= self.cooldown_s:
            return HALF_OPEN
        return OPEN

    def allow(self) -> bool:
        """May a request go to this model right now?

        Half-open admits exactly one probe at a time.
        """
        state = self.state
        if state == CLOSED:
            return True
        if state == HALF_OPEN and not self._probing:
            self._probing = True
            return True
        return False

    def would_allow(self) -> bool:
        """Side-effect-free peek used while planning (does not claim the probe)."""
        state = self.state
        return state == CLOSED or (state == HALF_OPEN and not self._probing)

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        self._probing = False
        self._failures += 1
        if self._opened_at is not None or self._failures >= self.failure_threshold:
            self._opened_at = self._clock.now()  # (re)open and restart cooldown
