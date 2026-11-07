"""Heuristic prompt-difficulty scoring.

The score decides the *minimum model quality* a prompt needs. It is a
transparent heuristic (no model call): every contribution is returned in
``reasons`` so a routing decision can be audited and the weights tuned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import estimate_tokens

_REASONING = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|analy[sz]e|compare and contrast|trade-?offs?|"
    r"optimi[sz]e|design|architect|refactor|debug|root cause|why does|explain why|"
    r"evaluate|critique)\b", re.I)
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|\bfunction\b|=>|#include|SELECT .* FROM|\bimport \w+", re.I | re.S)
_MATH = re.compile(r"\d\s*[\^*/=]\s*\d|\b(integral|derivative|matrix|probability|theorem|equation)\b", re.I)
_CONSTRAINT = re.compile(r"\b(must|exactly|at least|at most|do not|don't|without|only if|ensure that)\b", re.I)
_SIMPLE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|what is|what's|who is|define|translate|capital of|spell)\b", re.I)


@dataclass(frozen=True)
class Complexity:
    score: float                       # 0 (trivial) .. 1 (hard)
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def required_quality(self, floor: float = 0.35, ceiling: float = 0.95) -> float:
        """Map the score onto the quality a model must offer."""
        return round(floor + (ceiling - floor) * self.score, 4)


def score_prompt(prompt: str) -> Complexity:
    score = 0.0
    reasons: list[str] = []

    tokens = estimate_tokens(prompt)
    if tokens > 1500:
        score += 0.25; reasons.append(f"very long prompt (~{tokens} tokens) +0.25")
    elif tokens > 400:
        score += 0.15; reasons.append(f"long prompt (~{tokens} tokens) +0.15")
    elif tokens > 100:
        score += 0.07; reasons.append(f"medium prompt (~{tokens} tokens) +0.07")

    hits = len(_REASONING.findall(prompt))
    if hits:
        add = min(0.4, 0.2 * hits)
        score += add; reasons.append(f"{hits} reasoning cue(s) +{add:.2f}")
    if _CODE.search(prompt):
        score += 0.2; reasons.append("contains code +0.20")
    if _MATH.search(prompt):
        score += 0.15; reasons.append("contains math +0.15")
    constraints = len(_CONSTRAINT.findall(prompt))
    if constraints >= 2:
        add = min(0.15, 0.05 * constraints)
        score += add; reasons.append(f"{constraints} explicit constraints +{add:.2f}")
    steps = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", prompt))
    if steps >= 3:
        score += 0.1; reasons.append(f"{steps}-item multi-part request +0.10")

    if _SIMPLE.search(prompt) and tokens <= 40 and score == 0.0:
        reasons.append("short factual/greeting prompt")
    score = max(0.0, min(1.0, score))
    return Complexity(round(score, 4), tuple(reasons))
