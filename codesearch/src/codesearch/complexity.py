"""McCabe cyclomatic complexity per function/method chunk (AST based)."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def cyclomatic(node: ast.AST) -> int:
    """1 + number of decision points. Boolean operators add (operands - 1);
    each comprehension `for` and each `if` filter adds 1; match cases add 1 each
    (the wildcard case is not counted); `else` branches add nothing."""
    n = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
                              ast.ExceptHandler, ast.Assert)):
            n += 1
        elif isinstance(child, ast.BoolOp):
            n += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            n += 1 + len(child.ifs)
        elif isinstance(child, ast.match_case):
            wild = isinstance(child.pattern, ast.MatchAs) and child.pattern.pattern is None and child.guard is None
            n += 0 if wild else 1
    return n


def source_complexity(src: str) -> int | None:
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None
    return cyclomatic(tree)


def rank(chunks, root: Path, min_score: int = 1, include_tests: bool = False):
    """Return [(chunk, score)] for functions and methods, most complex first."""
    cache: dict[str, list[str]] = {}
    out = []
    for c in chunks:
        if c.kind not in ("function", "method") or (c.is_test and not include_tests):
            continue
        if c.path not in cache:
            try:
                cache[c.path] = (root / c.path).read_text(errors="replace").splitlines()
            except OSError:
                cache[c.path] = []
        lines = cache[c.path][c.start - 1:c.end]
        score = source_complexity("\n".join(lines)) if lines else None
        if score is not None and score >= min_score:
            out.append((c, score))
    out.sort(key=lambda t: (-t[1], t[0].path, t[0].start))
    return out
