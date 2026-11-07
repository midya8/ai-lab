"""Deterministic assertions over a model output."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    type: str
    passed: bool
    detail: str = ""


def _get_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(part)
    return cur


def _parse_json(output: str):
    s = output.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return json.loads(fence.group(1) if fence else s)


def run_check(check: dict, output: str) -> CheckResult:
    """Evaluate one check dict such as ``{"type": "contains", "value": "refund"}``.

    ``contains``/``not_contains``/``equals`` take ``ignore_case`` (default true
    for contains, false for equals). Unknown check types are a failed check
    rather than an exception, so one typo cannot hide the rest of a run.
    """
    t = check.get("type", "")
    try:
        if t in ("contains", "not_contains"):
            ic = check.get("ignore_case", True)
            hay, needle = (output.lower(), str(check["value"]).lower()) if ic else (output, str(check["value"]))
            found = needle in hay
            ok = found if t == "contains" else not found
            return CheckResult(t, ok, "" if ok else f"{'missing' if t == 'contains' else 'found'} {check['value']!r}")
        if t == "equals":
            a, b = output.strip(), str(check["value"]).strip()
            if check.get("ignore_case", False):
                a, b = a.lower(), b.lower()
            return CheckResult(t, a == b, "" if a == b else f"got {a[:60]!r}")
        if t == "regex":
            ok = re.search(check["value"], output, re.S | (re.I if check.get("ignore_case") else 0)) is not None
            return CheckResult(t, ok, "" if ok else f"no match for /{check['value']}/")
        if t == "max_words":
            n = len(output.split())
            return CheckResult(t, n <= check["value"], "" if n <= check["value"] else f"{n} words > {check['value']}")
        if t == "min_words":
            n = len(output.split())
            return CheckResult(t, n >= check["value"], "" if n >= check["value"] else f"{n} words < {check['value']}")
        if t == "max_chars":
            n = len(output)
            return CheckResult(t, n <= check["value"], "" if n <= check["value"] else f"{n} chars > {check['value']}")
        if t == "json_valid":
            _parse_json(output)
            return CheckResult(t, True)
        if t == "json_field":
            try:
                obj = _parse_json(output)
            except json.JSONDecodeError as e:
                return CheckResult(t, False, f"invalid JSON: {e.msg}")
            try:
                got = _get_path(obj, check["path"])
            except (KeyError, IndexError, ValueError):
                return CheckResult(t, False, f"no field {check['path']!r}")
            if "value" in check:
                ok = got == check["value"]
                return CheckResult(t, ok, "" if ok else f"{check['path']}={got!r}")
            return CheckResult(t, True)
    except json.JSONDecodeError as e:
        return CheckResult(t, False, f"invalid JSON: {e.msg}")
    except (KeyError, re.error, TypeError) as e:
        return CheckResult(t, False, f"bad check: {e!r}")
    return CheckResult(t or "?", False, "unknown check type")
