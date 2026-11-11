"""Refactoring hotspots: complex functions that many other chunks depend on.

risk = complexity * (1 + references). References are name-based (distinct other chunks that
mention the function's name), so dunder methods (`__init__` is mentioned everywhere) count 0 references and the score is a triage heuristic, not a measurement.
"""
from __future__ import annotations

from pathlib import Path

from .complexity import rank


def hotspots(chunks, root: Path, min_complexity: int = 5, include_tests: bool = False):
    """Return [(chunk, complexity, references, risk)] sorted by risk, highest first."""
    mentions: dict[str, set[int]] = {}
    for i, c in enumerate(chunks):
        if c.kind == "module" or (c.is_test and not include_tests):
            continue
        for n in set(c.ids):
            mentions.setdefault(n, set()).add(i)
    pos = {id(c): i for i, c in enumerate(chunks)}
    out = []
    for c, cx in rank(chunks, root, min_score=min_complexity, include_tests=include_tests):
        dunder = c.name.startswith("__") and c.name.endswith("__")
        refs = 0 if dunder else len(mentions.get(c.name, set()) - {pos[id(c)]})
        out.append((c, cx, refs, cx * (1 + refs)))
    out.sort(key=lambda t: (-t[3], t[0].id))
    return out
