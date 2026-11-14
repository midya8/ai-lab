"""Locate the sibling `microgrid` project (autodiff engine) and make it importable."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_mg = os.path.abspath(os.path.join(_here, "..", "..", "..", "microgrid", "src"))
if os.path.isdir(_mg) and _mg not in sys.path:
    sys.path.insert(0, _mg)

try:
    import microgrid  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("interplab needs the sibling microgrid project (ai-lab/microgrid/src) on the path") from e
