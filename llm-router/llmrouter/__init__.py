"""llm-router: cost/latency-aware routing across LLMs."""

from .breaker import CircuitBreaker
from .budget import Budget, BudgetExceeded
from .cache import ResponseCache, cache_key
from .clock import ManualClock, SystemClock
from .complexity import Complexity, score_prompt
from .models import ModelSpec, load_models
from .providers import Completion, ProviderError, SimProvider
from .router import STRATEGIES, AllModelsFailed, RouteResult, Router, RouterError

__all__ = [
    "AllModelsFailed", "Budget", "BudgetExceeded", "CircuitBreaker", "Complexity",
    "Completion", "ManualClock", "ModelSpec", "ProviderError", "ResponseCache",
    "RouteResult", "Router", "RouterError", "STRATEGIES", "SimProvider",
    "SystemClock", "cache_key", "load_models", "score_prompt",
]
__version__ = "0.1.0"
