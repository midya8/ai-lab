"""Model catalogue: pricing, latency, quality and context limits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


class CatalogueError(ValueError):
    """The model catalogue is malformed."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_cost_per_1k: float   # USD per 1,000 input tokens
    output_cost_per_1k: float  # USD per 1,000 output tokens
    latency_ms: float          # time to first token
    ms_per_token: float        # streaming speed
    quality: float             # 0..1, how hard a task it can be trusted with
    context_window: int        # max input tokens

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self.input_cost_per_1k
            + output_tokens / 1000 * self.output_cost_per_1k
        )

    def latency(self, output_tokens: int) -> float:
        return self.latency_ms + self.ms_per_token * output_tokens


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


def expected_output_tokens(prompt: str, max_tokens: int) -> int:
    """Planning estimate of the answer length, capped by ``max_tokens``."""
    return min(max_tokens, max(32, estimate_tokens(prompt)))


_FIELDS = (
    "name", "input_cost_per_1k", "output_cost_per_1k", "latency_ms",
    "ms_per_token", "quality", "context_window",
)


def parse_model(raw: dict) -> ModelSpec:
    missing = [f for f in _FIELDS if f not in raw]
    if missing:
        raise CatalogueError(f"model {raw.get('name', '?')!r} missing fields: {missing}")
    unknown = set(raw) - set(_FIELDS)
    if unknown:
        raise CatalogueError(f"model {raw['name']!r} has unknown fields: {sorted(unknown)}")
    spec = ModelSpec(
        name=str(raw["name"]),
        input_cost_per_1k=float(raw["input_cost_per_1k"]),
        output_cost_per_1k=float(raw["output_cost_per_1k"]),
        latency_ms=float(raw["latency_ms"]),
        ms_per_token=float(raw["ms_per_token"]),
        quality=float(raw["quality"]),
        context_window=int(raw["context_window"]),
    )
    if not spec.name:
        raise CatalogueError("model name must not be empty")
    if not 0.0 <= spec.quality <= 1.0:
        raise CatalogueError(f"{spec.name}: quality must be within 0..1")
    for f in ("input_cost_per_1k", "output_cost_per_1k", "latency_ms", "ms_per_token"):
        if getattr(spec, f) < 0:
            raise CatalogueError(f"{spec.name}: {f} must be >= 0")
    if spec.context_window <= 0:
        raise CatalogueError(f"{spec.name}: context_window must be > 0")
    return spec


def load_models(path: str | Path) -> list[ModelSpec]:
    data = json.loads(Path(path).read_text())
    raw = data.get("models") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise CatalogueError("catalogue must contain a non-empty 'models' list")
    models = [parse_model(m) for m in raw]
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise CatalogueError("duplicate model names in catalogue")
    return models
