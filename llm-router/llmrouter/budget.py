"""Rolling-window spend budget with a soft limit that triggers degradation."""

from __future__ import annotations

from collections import deque

from .clock import SystemClock


class BudgetExceeded(RuntimeError):
    """No affordable model is left inside the budget."""


class Budget:
    def __init__(self, limit_usd: float, window_s: float = 86400.0,
                 soft_ratio: float = 0.8, clock=None) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        if not 0 < soft_ratio <= 1:
            raise ValueError("soft_ratio must be within (0, 1]")
        self.limit_usd = limit_usd
        self.window_s = window_s
        self.soft_ratio = soft_ratio
        self._clock = clock or SystemClock()
        self._entries: deque[tuple[float, float]] = deque()

    def _expire(self) -> None:
        cutoff = self._clock.now() - self.window_s
        while self._entries and self._entries[0][0] <= cutoff:
            self._entries.popleft()

    @property
    def spent(self) -> float:
        self._expire()
        return sum(a for _, a in self._entries)

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent)

    @property
    def pressure(self) -> float:
        """Fraction of the limit already used (can exceed 1.0 only via record())."""
        return self.spent / self.limit_usd

    @property
    def degraded(self) -> bool:
        return self.pressure >= self.soft_ratio

    def can_afford(self, amount: float) -> bool:
        return amount <= self.remaining + 1e-12

    def record(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("amount must be >= 0")
        self._entries.append((self._clock.now(), amount))
