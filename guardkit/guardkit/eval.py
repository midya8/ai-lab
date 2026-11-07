"""Adversarial evaluation: measure the guard against labelled cases.

A case is one JSON object per line: ``{"id", "text", "label"}`` where
label is ``injection`` | ``secret`` | ``pii`` | ``benign``. The expected
outcome is derived from the label and the policy in use: attacks
(injection, secret) must block, PII must be caught (block or redact),
benign must be allowed. The report is precision/recall-style rates on
a small, honest fixture set — not a benchmark claim.

Case texts use only synthetic values (555 phone numbers, example.com
addresses, fake keys). Nothing real, nothing scraped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .guard import Guard
from .policy import Policy

__all__ = ["CaseResult", "EvalReport", "evaluate_cases", "load_cases"]

# What counts as "caught" for each label, given a decision action.
_CAUGHT = {
    "injection": {"block"},
    "secret": {"block"},
    "pii": {"block", "redact"},
    "benign": {"allow"},
}


@dataclass
class CaseResult:
    case_id: str
    label: str
    action: str
    score: float
    caught: bool
    expected: set[str]


@dataclass
class EvalReport:
    """Rates over the case set, plus the per-case outcomes."""

    total: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def attack_recall(self) -> float:
        attacks = [r for r in self.results if r.label in ("injection", "secret")]
        if not attacks:
            return 1.0
        return sum(r.caught for r in attacks) / len(attacks)

    @property
    def pii_caught_rate(self) -> float:
        pii = [r for r in self.results if r.label == "pii"]
        if not pii:
            return 1.0
        return sum(r.caught for r in pii) / len(pii)

    @property
    def false_positive_rate(self) -> float:
        benign = [r for r in self.results if r.label == "benign"]
        if not benign:
            return 0.0
        return sum(not r.caught for r in benign) / len(benign)

    def per_label(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for label in ("injection", "secret", "pii", "benign"):
            rows = [r for r in self.results if r.label == label]
            if not rows:
                continue
            out[label] = {
                "n": len(rows),
                "caught": sum(r.caught for r in rows),
                "rate": sum(r.caught for r in rows) / len(rows),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "attack_recall": round(self.attack_recall, 3),
            "pii_caught_rate": round(self.pii_caught_rate, 3),
            "false_positive_rate": round(self.false_positive_rate, 3),
            "per_label": {
                k: {"n": v["n"], "caught": v["caught"], "rate": round(v["rate"], 3)}
                for k, v in self.per_label().items()
            },
            "cases": [
                {
                    "id": r.case_id,
                    "label": r.label,
                    "action": r.action,
                    "score": round(r.score, 3),
                    "caught": r.caught,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# guardkit eval report",
            "",
            f"- Cases: {self.total}",
            f"- Attack recall (injection + secret blocked): {self.attack_recall:.3f}",
            f"- PII caught rate (blocked or redacted): {self.pii_caught_rate:.3f}",
            f"- False-positive rate (benign not allowed): {self.false_positive_rate:.3f}",
            "",
            "| Label | n | Caught | Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
        for label, row in self.per_label().items():
            lines.append(
                f"| {label} | {row['n']} | {row['caught']} | {row['rate']:.3f} |"
            )
        missed = [r for r in self.results if not r.caught]
        if missed:
            lines += ["", "## Misses", ""]
            for r in missed:
                lines.append(
                    f"- {r.case_id} ({r.label}): action={r.action}, "
                    f"expected one of {sorted(r.expected)}"
                )
        return "\n".join(lines) + "\n"


def load_cases(path: str) -> list[dict]:
    """Load a JSONL case file, validating ids, labels, and text."""
    cases: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from None
            if not isinstance(case, dict):
                raise ValueError(f"{path}:{lineno}: case must be an object")
            for key in ("id", "text", "label"):
                if key not in case:
                    raise ValueError(f"{path}:{lineno}: missing {key!r}")
            if case["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {case['id']!r}")
            if case["label"] not in _CAUGHT:
                raise ValueError(
                    f"{path}:{lineno}: unknown label {case['label']!r}"
                )
            seen.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def evaluate_cases(cases: list[dict], policy: Policy | None = None) -> EvalReport:
    """Run the guard over labelled cases and score the outcomes."""
    guard = Guard(policy)
    results: list[CaseResult] = []
    for case in cases:
        decision = guard.check_input(case["text"], dry_run=True)
        expected = _CAUGHT[case["label"]]
        results.append(
            CaseResult(
                case_id=case["id"],
                label=case["label"],
                action=decision.action,
                score=decision.score,
                caught=decision.action in expected,
                expected=set(expected),
            )
        )
    return EvalReport(total=len(results), results=results)
