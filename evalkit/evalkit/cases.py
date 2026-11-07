"""Loading and validation of evalkit golden sets.

A golden set is a JSON file (JSON is also valid YAML, so ``.yaml``/``.yml``
files containing JSON are accepted too) with this shape::

    {
      "version": 1,
      "name": "demo",
      "cases": [
        {
          "id": "sum-001",
          "prompt": "Summarise ...",
          "expected_contains": ["quarterly revenue"],
          "expected_not_contains": ["lorem ipsum"],
          "regex": ["^Subject: .+"],
          "json_valid": true,
          "length": {"min": 10, "max": 500},
          "rubric": [
            {"name": "mentions BPA-free", "contains": "BPA-free"}
          ],
          "tags": ["summarise"]
        }
      ]
    }

Every case must define at least one check, otherwise it cannot fail and
would silently inflate the score. Validation is strict on purpose: a bad
golden set is a bug, and it should fail loudly (exit code 2) before any
model call is made.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class CaseError(ValueError):
    """Raised when a golden-set file is invalid."""


@dataclass
class RubricItem:
    name: str
    contains: str | None = None
    regex: str | None = None


@dataclass
class Case:
    id: str
    prompt: str
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    regex: list[str] = field(default_factory=list)
    json_valid: bool = False
    length_min: int | None = None
    length_max: int | None = None
    rubric: list[RubricItem] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def check_count(self) -> int:
        """Number of individual checks this case will be scored on."""
        n = (
            len(self.expected_contains)
            + len(self.expected_not_contains)
            + len(self.regex)
            + len(self.rubric)
        )
        if self.json_valid:
            n += 1
        if self.length_min is not None or self.length_max is not None:
            n += 1
        return n


def _require_str_list(raw: dict, key: str, case_id: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CaseError(f"case {case_id!r}: {key!r} must be a list of strings")
    return list(value)


def _parse_case(raw: dict, index: int) -> Case:
    if not isinstance(raw, dict):
        raise CaseError(f"case at index {index} is not an object")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise CaseError(f"case at index {index} has no non-empty string 'id'")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CaseError(f"case {case_id!r}: 'prompt' must be a non-empty string")

    regexes = _require_str_list(raw, "regex", case_id)
    for pattern in regexes:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise CaseError(f"case {case_id!r}: invalid regex {pattern!r}: {exc}") from exc

    rubric_raw = raw.get("rubric", [])
    if not isinstance(rubric_raw, list):
        raise CaseError(f"case {case_id!r}: 'rubric' must be a list")
    rubric: list[RubricItem] = []
    for item in rubric_raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise CaseError(f"case {case_id!r}: rubric items need a string 'name'")
        contains = item.get("contains")
        pattern = item.get("regex")
        if contains is None and pattern is None:
            raise CaseError(
                f"case {case_id!r}: rubric item {item['name']!r} needs "
                "'contains' or 'regex'"
            )
        if contains is not None and not isinstance(contains, str):
            raise CaseError(f"case {case_id!r}: rubric 'contains' must be a string")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise CaseError(f"case {case_id!r}: rubric 'regex' must be a string")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise CaseError(
                    f"case {case_id!r}: invalid rubric regex {pattern!r}: {exc}"
                ) from exc
        rubric.append(RubricItem(name=item["name"], contains=contains, regex=pattern))

    length = raw.get("length", None)
    length_min = length_max = None
    if length is not None:
        if not isinstance(length, dict):
            raise CaseError(f"case {case_id!r}: 'length' must be an object")
        length_min = length.get("min")
        length_max = length.get("max")
        for bound in (length_min, length_max):
            if bound is not None and (not isinstance(bound, int) or bound < 0):
                raise CaseError(
                    f"case {case_id!r}: length bounds must be non-negative integers"
                )
        if (
            length_min is not None
            and length_max is not None
            and length_min > length_max
        ):
            raise CaseError(
                f"case {case_id!r}: length min ({length_min}) > max ({length_max})"
            )

    json_valid = raw.get("json_valid", False)
    if not isinstance(json_valid, bool):
        raise CaseError(f"case {case_id!r}: 'json_valid' must be a boolean")

    case = Case(
        id=case_id,
        prompt=prompt,
        expected_contains=_require_str_list(raw, "expected_contains", case_id),
        expected_not_contains=_require_str_list(raw, "expected_not_contains", case_id),
        regex=regexes,
        json_valid=json_valid,
        length_min=length_min,
        length_max=length_max,
        rubric=rubric,
        tags=_require_str_list(raw, "tags", case_id),
    )
    if case.check_count() == 0:
        raise CaseError(
            f"case {case_id!r}: defines no checks — add at least one of "
            "expected_contains, expected_not_contains, regex, json_valid, "
            "length, or rubric"
        )
    return case


def load_cases(path: str | Path) -> list[Case]:
    """Load and validate a golden-set file. Raises CaseError on any problem."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CaseError(f"cannot read golden set {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CaseError(f"golden set {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise CaseError(f"golden set {path} must be an object with a 'cases' list")
    if not data["cases"]:
        raise CaseError(f"golden set {path} contains no cases")

    cases: list[Case] = []
    seen: set[str] = set()
    for index, raw in enumerate(data["cases"]):
        case = _parse_case(raw, index)
        if case.id in seen:
            raise CaseError(f"duplicate case id {case.id!r} in {path}")
        seen.add(case.id)
        cases.append(case)
    return cases


def filter_cases(
    cases: list[Case], only_tag: str | None = None, id_substring: str | None = None
) -> list[Case]:
    """Select cases by tag and/or id substring (both must match if given)."""
    selected = cases
    if only_tag is not None:
        selected = [c for c in selected if only_tag in c.tags]
    if id_substring is not None:
        selected = [c for c in selected if id_substring in c.id]
    return selected
