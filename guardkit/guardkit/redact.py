"""Redaction with a reversible, in-memory token vault.

Detected spans are replaced by placeholder tokens (``[EMAIL_1]``,
``[CARD_1]`` ...) so a prompt can still be sent to a model — or logged —
without the raw value leaving the process. The vault lives only in
memory: it is never written to the audit log, never serialised, and
``restore`` only puts back values that this vault issued, for the same
conversation. If the process exits, the mapping is gone.
"""

from __future__ import annotations

from .detectors import Finding

__all__ = ["RedactionVault", "redact_spans"]


class RedactionVault:
    """Maps placeholder tokens back to the values they replaced."""

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, finding: Finding) -> str:
        """Issue (or reuse) a token for this finding's span."""
        key = finding.rule_id
        n = self._counters.get(key, 0) + 1
        self._counters[key] = n
        token = f"[{key.upper()}_{n}]"
        self._tokens[token] = finding.matched
        return token

    def restore(self, text: str) -> str:
        """Put original values back wherever a vault token appears."""
        for token, value in self._tokens.items():
            text = text.replace(token, value)
        return text

    def __len__(self) -> int:
        return len(self._tokens)


def redact_spans(
    text: str, findings: list[Finding], vault: RedactionVault | None = None
) -> tuple[str, dict[str, str]]:
    """Replace finding spans with tokens.

    Returns (redacted_text, {token: original}) for the spans replaced.
    Overlapping findings are resolved by keeping the earliest, longest
    span; a finding with an empty span is skipped. With no vault, the
    mapping is still returned so the caller can decide what to keep —
    but the caller must not log it.
    """
    if vault is None:
        vault = RedactionVault()
    chosen: list[Finding] = []
    for f in sorted(findings, key=lambda x: (x.start, -(x.end - x.start))):
        if f.end <= f.start:
            continue
        if any(f.start < c.end and c.start < f.end for c in chosen):
            continue
        chosen.append(f)
    out: list[str] = []
    mapping: dict[str, str] = {}
    pos = 0
    for f in sorted(chosen, key=lambda x: x.start):
        out.append(text[pos : f.start])
        token = vault.token_for(f)
        mapping[token] = f.matched
        out.append(token)
        pos = f.end
    out.append(text[pos:])
    return "".join(out), mapping
