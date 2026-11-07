"""Deterministic detectors for PII, secrets, and prompt-injection signals.

Everything here is a plain, inspectable pattern check — no model
judgement, no network, no heuristics that need tuning data. A finding
names the rule, the category, the severity, and the exact span, so a
caller can redact it, block on it, or log it without copying the raw
value into a report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Finding",
    "InjectionScorer",
    "PiiSecretScanner",
    "INJECTION_SIGNALS",
    "scan_text",
]

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One detector hit: what matched, where, and how bad it is."""

    rule_id: str
    category: str  # "pii" | "secret" | "injection"
    severity: str  # "low" | "medium" | "high" | "critical"
    start: int
    end: int
    message: str
    matched: str = ""
    weight: float = 0.0  # injection signals only: contribution to the score

    def redacted_excerpt(self, text: str, context: int = 24) -> str:
        """Excerpt with the matched span replaced by its rule token."""
        lo = max(0, self.start - context)
        hi = min(len(text), self.end + context)
        return (
            text[lo : self.start]
            + f"[{self.rule_id.upper()}]"
            + text[self.end : hi]
        )


# ---------------------------------------------------------------------------
# PII and secrets
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
)
_CARD_CANDIDATE_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){13,19}\d(?![\d-])")
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)
_ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?token|access[_-]?token|secret|password|passwd)"
    r"\s*[:=]\s*['\"]?([^\s'\";]{8,})"
)


def luhn_ok(digits: str) -> bool:
    """True when the digit string passes the Luhn checksum."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_phone_candidate(text: str, start: int, end: int) -> bool:
    """Reject digit runs that are really card numbers or long ids."""
    digits = re.sub(r"\D", "", text[start:end])
    return 10 <= len(digits) <= 11 and not (
        len(digits) >= 13 and luhn_ok(digits)
    )


@dataclass
class PiiSecretScanner:
    """Finds PII and secret material in a string.

    Card numbers are validated with Luhn, so a random 16-digit run is
    not flagged. Phone numbers only match the North-American shape;
    anything else is left alone rather than guessed at.
    """

    enabled: tuple[str, ...] = (
        "email",
        "phone",
        "card",
        "jwt",
        "private_key",
        "assigned_secret",
    )

    def scan(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        enabled = set(self.enabled)
        if "email" in enabled:
            for m in _EMAIL_RE.finditer(text):
                findings.append(
                    Finding(
                        "pii-email",
                        "pii",
                        "medium",
                        m.start(),
                        m.end(),
                        "email address",
                        m.group(0),
                    )
                )
        if "card" in enabled:
            for m in _CARD_CANDIDATE_RE.finditer(text):
                digits = re.sub(r"\D", "", m.group(0))
                if 13 <= len(digits) <= 19 and luhn_ok(digits):
                    findings.append(
                        Finding(
                            "pii-card",
                            "pii",
                            "critical",
                            m.start(),
                            m.end(),
                            "payment-card number (Luhn-valid)",
                            m.group(0),
                        )
                    )
        if "phone" in enabled:
            for m in _PHONE_RE.finditer(text):
                if _is_phone_candidate(text, m.start(), m.end()):
                    # Skip spans already claimed by a card finding.
                    if any(
                        f.start <= m.start() and m.end() <= f.end
                        for f in findings
                    ):
                        continue
                    findings.append(
                        Finding(
                            "pii-phone",
                            "pii",
                            "medium",
                            m.start(),
                            m.end(),
                            "phone number",
                            m.group(0),
                        )
                    )
        if "jwt" in enabled:
            for m in _JWT_RE.finditer(text):
                findings.append(
                    Finding(
                        "secret-jwt",
                        "secret",
                        "high",
                        m.start(),
                        m.end(),
                        "JSON Web Token",
                        m.group(0),
                    )
                )
        if "private_key" in enabled:
            for m in _PRIVATE_KEY_RE.finditer(text):
                findings.append(
                    Finding(
                        "secret-private-key",
                        "secret",
                        "critical",
                        m.start(),
                        m.end(),
                        "private key header",
                        m.group(0),
                    )
                )
        if "assigned_secret" in enabled:
            for m in _ASSIGNED_SECRET_RE.finditer(text):
                findings.append(
                    Finding(
                        "secret-assigned",
                        "secret",
                        "high",
                        m.start(),
                        m.end(),
                        f"assigned secret ({m.group(1).lower()})",
                        m.group(0),
                    )
                )
        findings.sort(key=lambda f: (f.start, f.end))
        return findings


# ---------------------------------------------------------------------------
# Prompt-injection signals
# ---------------------------------------------------------------------------

# Each signal is (rule_id, weight, [regexes]). Weights sum and cap at 1.0;
# a single strong signal is enough to cross the default 0.5 threshold.
INJECTION_SIGNALS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    (
        "injection-ignore-instructions",
        0.50,
        (
            r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions",
            r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions",
            r"forget\s+(?:your|all|the)\s+instructions",
        ),
    ),
    (
        "injection-reveal-system",
        0.50,
        (
            r"reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions|prompt)",
            r"print\s+(?:your|the)\s+system\s+prompt",
            r"repeat\s+(?:your|the)\s+(?:system\s+)?instructions",
            r"what\s+(?:are|is)\s+your\s+(?:system\s+)?instructions",
            r"show\s+(?:me\s+)?(?:your|the)\s+system\s+prompt",
        ),
    ),
    (
        "injection-role-confusion",
        0.40,
        (
            r"you\s+are\s+now\s+(?:a|an|in)\s+",
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|uncensored)",
            r"pretend\s+you\s+are\s+(?:a|an|not)",
            r"\bDAN\b",
        ),
    ),
    (
        "injection-exfiltrate",
        0.50,
        (
            r"exfiltrate\b",
            r"send\s+(?:the|all|any|your)\s+(?:api\s+key|secret|password|credentials?|system\s+prompt)",
            r"leak\s+(?:the|your)\s+(?:api\s+key|secret|password|system\s+prompt)",
            r"output\s+(?:the|your)\s+(?:api\s+key|secret|password)",
        ),
    ),
    (
        "injection-bypass",
        0.45,
        (
            r"bypass\s+(?:the|all|any)\s+(?:filter|guardrail|safety|restriction|content\s+policy)",
            r"disable\s+(?:the|your)\s+(?:safety|filter|guardrail|content\s+policy)",
        ),
    ),
    (
        "injection-delimiter",
        0.35,
        (
            r"```\s*system\b",
            r"\[SYSTEM\]",
            r"<\|system\|>",
            r"---\s*system\s+instructions\s*:",
        ),
    ),
    (
        "injection-encoded",
        0.30,
        (
            r"base64\s+decode\b",
            r"decode\s+this\s+base64",
        ),
    ),
    (
        "injection-override",
        0.30,
        (
            r"override\s+(?:your|the)\s+instructions",
            r"new\s+instructions\s*:",
        ),
    ),
)


@dataclass
class InjectionScorer:
    """Scores text for prompt-injection language.

    Returns a 0.0–1.0 score plus the signals that fired. Deterministic
    pattern checks: a score is a sum of fired-signal weights, capped
    at 1.0. It is a fast gate, not a claim that injection is caught in
    general — adversarial phrasing that avoids these patterns will
    score low, and that limitation is stated in the README.
    """

    signals: tuple = INJECTION_SIGNALS

    def score(self, text: str) -> tuple[float, list[Finding]]:
        findings: list[Finding] = []
        total = 0.0
        for rule_id, weight, patterns in self.signals:
            for pattern in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    total += weight
                    findings.append(
                        Finding(
                            rule_id,
                            "injection",
                            "high",
                            m.start(),
                            m.end(),
                            f"injection signal: {rule_id}",
                            m.group(0),
                            weight=weight,
                        )
                    )
                    break  # one hit per signal family
        return min(1.0, total), findings


def scan_text(text: str) -> list[Finding]:
    """All PII/secret findings plus injection-signal findings."""
    findings = PiiSecretScanner().scan(text)
    _, injection = InjectionScorer().score(text)
    return sorted(findings + injection, key=lambda f: (f.start, f.end))
