"""A small Mongo-style metadata filter language.

    {"lang": "en"}                          equality
    {"year": {"$gte": 2020, "$lt": 2024}}   comparisons: $eq $ne $gt $gte $lt $lte
    {"tag": {"$in": ["a", "b"]}}            membership: $in $nin
    {"$and": [f1, f2]}, {"$or": [...]}, {"$not": f}
    several keys in one dict are ANDed.

A missing field never matches a comparison or ``$in`` (and always matches ``$ne``/``$nin``);
comparing incompatible types (e.g. a string against an int) is simply "no match", not an error.
"""
from __future__ import annotations

from typing import Any, Callable

_CMP = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a > b,
    "$gte": lambda a, b: a >= b,
    "$lt": lambda a, b: a < b,
    "$lte": lambda a, b: a <= b,
}
_MISSING = object()


def compile_filter(spec: dict) -> Callable[[dict], bool]:
    """Validate ``spec`` once and return a predicate over a metadata dict."""
    if not isinstance(spec, dict):
        raise ValueError(f"filter must be a dict, got {type(spec).__name__}")
    preds = [_compile_clause(k, v) for k, v in spec.items()]

    def match(meta: dict) -> bool:
        return all(p(meta) for p in preds)

    return match


def _compile_clause(key: str, val: Any) -> Callable[[dict], bool]:
    if key in ("$and", "$or"):
        if not isinstance(val, list) or not val:
            raise ValueError(f"{key} needs a non-empty list of filters")
        subs = [compile_filter(v) for v in val]
        if key == "$and":
            return lambda m: all(s(m) for s in subs)
        return lambda m: any(s(m) for s in subs)
    if key == "$not":
        sub = compile_filter(val)
        return lambda m: not sub(m)
    if key.startswith("$"):
        raise ValueError(f"unknown operator {key!r}")
    if isinstance(val, dict) and val and all(k.startswith("$") for k in val):
        ops = [_compile_op(op, arg) for op, arg in val.items()]
        return lambda m: all(o(m.get(key, _MISSING)) for o in ops)
    return lambda m: m.get(key, _MISSING) == val


def _compile_op(op: str, arg: Any) -> Callable[[Any], bool]:
    if op in _CMP:
        f = _CMP[op]

        def cmp(v):
            if v is _MISSING:
                return op == "$ne"
            try:
                return bool(f(v, arg))
            except TypeError:
                return op == "$ne"
        return cmp
    if op in ("$in", "$nin"):
        if not isinstance(arg, (list, tuple, set)):
            raise ValueError(f"{op} needs a list")
        pool = list(arg)

        def member(v):
            if v is _MISSING:
                return op == "$nin"
            return (v in pool) == (op == "$in")
        return member
    raise ValueError(f"unknown operator {op!r}")
