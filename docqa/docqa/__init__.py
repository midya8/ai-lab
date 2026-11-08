"""docqa: grounded document Q&A with citations and abstention."""
import sys
from pathlib import Path

try:  # prefer an installed ragkit, fall back to the sibling folder in the monorepo
    import ragkit  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ragkit"))

from .qa import DocQA, QAResult, Evidence  # noqa: E402

__all__ = ["DocQA", "QAResult", "Evidence"]
