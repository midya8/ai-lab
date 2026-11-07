"""Deterministic scorers for evalkit.

Every check is a plain, inspectable rule — no model calls, no hidden
judgement. A case's score is the fraction of its checks that pass, so a
score of 1.0 means every check passed and anything below 1.0 is a
(partial) failure. ``passed`` is True only when the score is exactly 1.0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .cases import Case


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseScore:
    case_id: str
    score: float
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _contains_check(needle: str, response: str) -> CheckResult:
    found = needle in response
    return CheckResult(
        name=f"contains:{needle!r}",
        passed=found,
        detail=f"found in response" if found else f"missing from response",
    )


def _not_contains_check(needle: str, response: str) -> CheckResult:
    absent = needle not in response
    return CheckResult(
        name=f"not_contains:{needle!r}",
        passed=absent,
        detail="absent from response" if absent else "present in response (must not be)",
    )


def _regex_check(pattern: str, response: str) -> CheckResult:
    matched = re.search(pattern, response) is not None
    return CheckResult(
        name=f"regex:{pattern!r}",
        passed=matched,
        detail="matched" if matched else "no match",
    )


def _json_valid_check(response: str) -> CheckResult:
    try:
        json.loads(response)
    except json.JSONDecodeError as exc:
        return CheckResult(
            name="json_valid", passed=False, detail=f"invalid JSON: {exc}"
        )
    return CheckResult(name="json_valid", passed=True, detail="valid JSON")


def _length_check(case: Case, response: str) -> CheckResult:
    n = len(response)
    lo = case.length_min if case.length_min is not None else 0
    hi = case.length_max if case.length_max is not None else n
    ok = lo <= n <= hi
    return CheckResult(
        name=f"length:{lo}..{hi}",
        passed=ok,
        detail=f"response is {n} chars",
    )


def score_response(case: Case, response: str) -> CaseScore:
    """Score one model response against one case. Never raises on bad input."""
    checks: list[CheckResult] = []
    for needle in case.expected_contains:
        checks.append(_contains_check(needle, response))
    for needle in case.expected_not_contains:
        checks.append(_not_contains_check(needle, response))
    for pattern in case.regex:
        checks.append(_regex_check(pattern, response))
    if case.json_valid:
        checks.append(_json_valid_check(response))
    if case.length_min is not None or case.length_max is not None:
        checks.append(_length_check(case, response))
    for item in case.rubric:
        if item.contains is not None:
            result = _contains_check(item.contains, response)
        else:
            result = _regex_check(item.regex or "", response)
        result.name = f"rubric:{item.name}"
        checks.append(result)

    if not checks:
        # A case with no checks cannot fail; validation normally prevents
        # this, but the scorer stays total so library callers are safe.
        return CaseScore(case_id=case.id, score=1.0, passed=True, checks=checks)
    passed = sum(1 for c in checks if c.passed)
    score = passed / len(checks)
    return CaseScore(
        case_id=case.id, score=score, passed=score == 1.0, checks=checks
    )
