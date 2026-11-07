"""Provider interface and a deterministic simulated provider.

A provider has one method, ``complete(model, prompt, max_tokens)``, and
raises :class:`ProviderError` for failures the router may recover from.
:class:`SimProvider` needs no network or API key and is what the tests,
the simulator and the CI demo use.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Protocol

from .clock import ManualClock
from .models import ModelSpec, estimate_tokens, expected_output_tokens


class ProviderError(RuntimeError):
    """A model call failed (timeout, 5xx, rate limit...)."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class Provider(Protocol):
    def complete(self, model: ModelSpec, prompt: str, max_tokens: int) -> Completion: ...


class SimProvider:
    """Seeded fake provider with per-model failure rates and outages."""

    def __init__(self, failure_rates: dict[str, float] | None = None,
                 seed: int = 0, clock: ManualClock | None = None) -> None:
        self.failure_rates = dict(failure_rates or {})
        self.down: set[str] = set()
        self.calls: list[str] = []
        self._rng = random.Random(seed)
        self._clock = clock

    def complete(self, model: ModelSpec, prompt: str, max_tokens: int) -> Completion:
        self.calls.append(model.name)
        if model.name in self.down:
            raise ProviderError(f"{model.name}: simulated outage")
        if self._rng.random() < self.failure_rates.get(model.name, 0.0):
            raise ProviderError(f"{model.name}: simulated transient failure")
        in_tok = estimate_tokens(prompt)
        out_tok = expected_output_tokens(prompt, max_tokens)
        latency = model.latency(out_tok)
        if self._clock is not None:
            self._clock.advance(latency / 1000)
        digest = hashlib.sha256(f"{model.name}|{prompt}".encode()).hexdigest()[:8]
        return Completion(f"[{model.name}:{digest}] simulated answer", in_tok, out_tok, latency)
