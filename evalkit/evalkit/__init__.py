"""evalkit — a small, dependency-free LLM evaluation harness.

Run a versioned golden set of prompts against a pluggable model adapter,
score each response with deterministic checks, and diff runs to catch
prompt/model regressions before they ship.
"""

from .cases import Case, CaseError, RubricItem, load_cases
from .runner import CaseRun, RunResult, run_suite
from .scorers import CaseScore, CheckResult, score_response

__version__ = "0.1.0"

__all__ = [
    "Case",
    "CaseError",
    "CaseRun",
    "CaseScore",
    "CheckResult",
    "RubricItem",
    "RunResult",
    "load_cases",
    "run_suite",
    "score_response",
    "__version__",
]
