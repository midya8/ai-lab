"""Light intra-function taint analysis for Python.

Tracks values from obvious untrusted sources (``input()``, ``sys.argv``,
``request.args/form/json/...``) through assignments, f-strings, concatenation,
``%``/``.format`` into dangerous sinks (SQL ``execute``, shell commands,
``eval``/``exec``). It is flow-insensitive within a function body processed in
source order, has no inter-procedural reasoning, and only treats a name as
clean when it is reassigned from a clean expression. Function parameters are
NOT assumed tainted (that would be too noisy); only explicit sources are.
"""
from __future__ import annotations

import ast
from typing import Iterator, List, Set, Tuple

_REQUEST_ATTRS = {"args", "form", "json", "values", "cookies", "data", "files", "headers", "GET", "POST", "body"}
_SOURCE_CALLS = {"input", "raw_input"}
_SQL_METHODS = {"execute", "executemany", "executescript"}
_SUBPROCESS = {"run", "call", "check_call", "check_output", "Popen"}
_SANITIZERS = {"int", "float", "bool", "len", "shlex.quote", "quote"}

Hit = Tuple[int, str, str]  # (line, rule, message)


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _is_source(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name in _SOURCE_CALLS:
            return True
        if name in {"request.get_json", "request.args.get", "request.form.get", "request.values.get"}:
            return True
    d = _dotted(node)
    if d == "sys.argv":
        return True
    parts = d.split(".")
    return len(parts) >= 2 and parts[0] == "request" and parts[1] in _REQUEST_ATTRS


def tainted(node: ast.AST, env: Set[str]) -> bool:
    if node is None:
        return False
    if _is_source(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in env
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name in _SANITIZERS:
            return False
        if isinstance(node.func, ast.Attribute) and tainted(node.func.value, env):
            return True  # method on tainted value (e.g. s.strip(), "..".format is handled below)
        return any(tainted(a, env) for a in node.args) or any(tainted(k.value, env) for k in node.keywords)
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        return tainted(node.value, env)
    if isinstance(node, ast.BinOp):
        return tainted(node.left, env) or tainted(node.right, env)
    if isinstance(node, ast.JoinedStr):
        return any(tainted(v, env) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return tainted(node.value, env)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(tainted(e, env) for e in node.elts)
    if isinstance(node, ast.IfExp):
        return tainted(node.body, env) or tainted(node.orelse, env)
    if isinstance(node, ast.BoolOp):
        return any(tainted(v, env) for v in node.values)
    return False


def _builds_string(node: ast.AST) -> bool:
    """True when the expression composes a string (f-string, +, %, .format), as opposed to a bare name."""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"


def _targets(t: ast.AST) -> Iterator[str]:
    if isinstance(t, ast.Name):
        yield t.id
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _targets(e)


def _own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """Walk a function body in source order without descending into nested functions."""
    stack = list(reversed(getattr(fn, "body", [])))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(n))))


def _check_sink(call: ast.Call, env: Set[str]) -> List[Hit]:
    name = _dotted(call.func)
    attr = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    hits: List[Hit] = []
    if attr in _SQL_METHODS and call.args:
        q = call.args[0]
        if tainted(q, env) and (_builds_string(q) or isinstance(q, ast.Name)):
            hits.append((call.lineno, "py-taint-sql",
                         "Untrusted input reaches a SQL `execute` as part of the query text (SQL injection)."))
    elif name in {"os.system", "os.popen"} and call.args and tainted(call.args[0], env):
        hits.append((call.lineno, "py-taint-shell", f"Untrusted input reaches `{name}` (command injection)."))
    elif name.startswith("subprocess.") and name.split(".")[-1] in _SUBPROCESS and call.args:
        shell = any(k.arg == "shell" and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords)
        first = call.args[0]
        if shell and tainted(first, env):
            hits.append((call.lineno, "py-taint-shell", f"Untrusted input reaches `{name}(..., shell=True)` (command injection)."))
    elif name in {"eval", "exec"} and call.args and tainted(call.args[0], env):
        hits.append((call.lineno, "py-taint-eval", f"Untrusted input reaches `{name}` (arbitrary code execution)."))
    return hits


def analyze_function(fn: ast.AST) -> List[Hit]:
    env: Set[str] = set()
    hits: List[Hit] = []
    for n in _own_nodes(fn):
        if isinstance(n, ast.Assign):
            t = tainted(n.value, env)
            for tgt in n.targets:
                for nm in _targets(tgt):
                    (env.add if t else env.discard)(nm)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            if tainted(n.value, env):
                env.add(n.target.id)
        elif isinstance(n, ast.AnnAssign) and n.value is not None and isinstance(n.target, ast.Name):
            (env.add if tainted(n.value, env) else env.discard)(n.target.id)
        elif isinstance(n, ast.For):
            if tainted(n.iter, env):
                env.update(_targets(n.target))
        elif isinstance(n, ast.Call):
            hits.extend(_check_sink(n, env))
    return hits


def taint_hits(tree: ast.AST) -> List[Hit]:
    """Analyse every function plus module level; returns de-duplicated (line, rule, message)."""
    seen: Set[Tuple[int, str]] = set()
    out: List[Hit] = []
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for sc in scopes:
        for h in analyze_function(sc):
            if (h[0], h[1]) not in seen:
                seen.add((h[0], h[1]))
                out.append(h)
    return sorted(out)
