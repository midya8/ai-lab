"""Cost metering: wrap any model callable, count tokens, price them, enforce a budget.

Token counts are *estimates* (``len(text) / 4`` rounded up, a common rule of
thumb for English) unless you pass your own ``count`` function, e.g. a real
tokenizer. Prices are whatever you give it, in dollars per million tokens;
nothing here knows any vendor's current price list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised *before* a call that would push spend over the budget. Aborts a run (never counted as a model failure)."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def parse_price(spec: str) -> tuple[float, float]:
    """``"IN,OUT"`` dollars per million tokens, e.g. ``"0.15,0.60"``."""
    try:
        a, b = spec.split(",")
        pin, pout = float(a), float(b)
    except ValueError:
        raise ValueError(f"bad price {spec!r}: use IN,OUT in dollars per million tokens, e.g. 0.15,0.60") from None
    if pin < 0 or pout < 0 or not math.isfinite(pin + pout):
        raise ValueError("prices must be finite and non-negative")
    return pin, pout


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0


class CostMeter:
    """Callable wrapper: ``CostMeter(model, price_in, price_out, budget=None)``.

    The budget check uses the prompt's cost plus an output allowance
    (``max_out_tokens``, default 0, so a call is only refused once the input
    alone would exceed it). Spend after a call can therefore overshoot by at
    most one reply. Failed calls still bill their input tokens.
    """

    def __init__(self, model, price_in: float, price_out: float, budget: float | None = None,
                 count=estimate_tokens, max_out_tokens: int = 0):
        self.model, self.price_in, self.price_out = model, price_in, price_out
        self.budget, self.count, self.max_out = budget, count, max_out_tokens
        self.usage = Usage()

    def _price(self, tin: int, tout: int) -> float:
        return (tin * self.price_in + tout * self.price_out) / 1e6

    def __call__(self, prompt: str) -> str:
        tin = self.count(prompt)
        if self.budget is not None and self.usage.cost + self._price(tin, self.max_out) > self.budget + 1e-12:
            raise BudgetExceeded(f"budget ${self.budget:.4f} reached after {self.usage.calls} call(s) "
                                 f"(spent ${self.usage.cost:.4f})")
        self.usage.calls += 1
        self.usage.tokens_in += tin
        self.usage.cost += self._price(tin, 0)
        out = self.model(prompt)
        tout = self.count(out)
        self.usage.tokens_out += tout
        self.usage.cost += self._price(0, tout)
        return out

    def summary(self) -> str:
        u = self.usage
        return (f"cost: ${u.cost:.4f} over {u.calls} call(s) "
                f"({u.tokens_in} in / {u.tokens_out} out tokens, estimated)")
