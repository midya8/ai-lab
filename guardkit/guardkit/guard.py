"""The Guard: one object that decides allow / redact / block.

``check_input`` guards a prompt before it reaches a model; ``check_output``
guards a model response before it reaches a user (leak scan only — a
model quoting an attack is not itself an attack). ``wrap`` composes both
around any ``str -> str`` callable, restoring redacted values the model
echoes back as placeholders so the user sees their own data, not tokens.

Decisions are fail-closed: oversize input, injection at or above the
threshold, and secrets block. PII redacts by default. Nothing here
touches the network and nothing needs a key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audit import AuditLog
from .detectors import Finding, InjectionScorer, PiiSecretScanner
from .policy import Policy, default_policy
from .redact import RedactionVault, redact_spans

__all__ = ["Decision", "Guard"]


@dataclass
class Decision:
    """The outcome of one guard check."""

    action: str  # "allow" | "redact" | "block"
    safe_text: str  # text safe to forward (redacted, or "" when blocked)
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0  # injection score, 0.0–1.0
    reasons: list[str] = field(default_factory=list)
    vault: RedactionVault | None = None

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def public_findings(self) -> list[dict]:
        """Findings safe to show or log: no raw matched values."""
        return [
            {
                "rule_id": f.rule_id,
                "category": f.category,
                "severity": f.severity,
                "message": f.message,
            }
            for f in self.findings
        ]


class Guard:
    """Runtime input/output guardrails driven by a Policy."""

    def __init__(
        self,
        policy: Policy | None = None,
        audit: AuditLog | None = None,
    ):
        self.policy = policy or default_policy()
        self.audit = audit
        self._scanner = PiiSecretScanner(
            enabled=tuple(
                d
                for d in self.policy.enabled_detectors
                if d != "injection"
            )
        )
        self._scorer = InjectionScorer()

    # -- input -----------------------------------------------------------

    def check_input(self, text: str, *, dry_run: bool = False) -> Decision:
        vault = RedactionVault()
        findings: list[Finding] = []
        reasons: list[str] = []

        if len(text) > self.policy.max_input_chars:
            decision = Decision(
                "block",
                "",
                findings,
                0.0,
                [f"input exceeds max_input_chars ({self.policy.max_input_chars})"],
                vault,
            )
            self._record("in", decision, text, dry_run)
            return decision

        score, injection = (
            self._scorer.score(text)
            if "injection" in self.policy.enabled_detectors
            else (0.0, [])
        )
        findings.extend(injection)
        pii_secret = self._scanner.scan(text)
        findings.extend(pii_secret)
        findings.sort(key=lambda f: (f.start, f.end))

        secrets = [f for f in findings if f.category == "secret"]
        pii = [f for f in findings if f.category == "pii"]

        if score >= self.policy.injection_threshold:
            reasons.append(
                f"injection score {score:.2f} >= threshold "
                f"{self.policy.injection_threshold:.2f}"
            )
            if self.policy.injection_action == "block":
                decision = Decision("block", "", findings, score, reasons, vault)
                self._record("in", decision, text, dry_run)
                return decision
            reasons.append("injection allowed by policy (warn-only)")

        if secrets and self.policy.secret_action == "block":
            reasons.append(f"secret material found: {sorted({f.rule_id for f in secrets})}")
            decision = Decision("block", "", findings, score, reasons, vault)
            self._record("in", decision, text, dry_run)
            return decision

        redactable = list(pii)
        if secrets and self.policy.secret_action == "redact":
            redactable.extend(secrets)
            reasons.append("secret material redacted per policy")
        if pii and self.policy.pii_action == "redact":
            reasons.append(f"PII redacted: {sorted({f.rule_id for f in pii})}")
        if pii and self.policy.pii_action == "block":
            reasons.append(f"PII found, policy blocks: {sorted({f.rule_id for f in pii})}")
            decision = Decision("block", "", findings, score, reasons, vault)
            self._record("in", decision, text, dry_run)
            return decision

        if redactable:
            safe_text, _ = redact_spans(text, redactable, vault)
            decision = Decision("redact", safe_text, findings, score, reasons, vault)
        else:
            decision = Decision("allow", text, findings, score, reasons, vault)
        self._record("in", decision, text, dry_run)
        return decision

    # -- output ----------------------------------------------------------

    def check_output(self, text: str, *, dry_run: bool = False) -> Decision:
        """Leak scan on model output: secrets block, PII redacts.

        Injection scoring is deliberately not applied here — output
        that quotes an attack is a leak problem, not an injection.
        """
        vault = RedactionVault()
        findings = self._scanner.scan(text)
        secrets = [f for f in findings if f.category == "secret"]
        pii = [f for f in findings if f.category == "pii"]
        reasons: list[str] = []

        if secrets and self.policy.secret_action == "block":
            reasons.append(
                f"model output leaks secret material: {sorted({f.rule_id for f in secrets})}"
            )
            decision = Decision("block", "", findings, 0.0, reasons, vault)
            self._record("out", decision, text, dry_run)
            return decision

        redactable = list(pii)
        if secrets and self.policy.secret_action == "redact":
            redactable.extend(secrets)
            reasons.append("secret material in output redacted per policy")
        if pii and self.policy.pii_action == "redact":
            reasons.append(f"PII in output redacted: {sorted({f.rule_id for f in pii})}")
        if pii and self.policy.pii_action == "block":
            reasons.append("PII in output, policy blocks")
            decision = Decision("block", "", findings, 0.0, reasons, vault)
            self._record("out", decision, text, dry_run)
            return decision

        if redactable:
            safe_text, _ = redact_spans(text, redactable, vault)
            decision = Decision("redact", safe_text, findings, 0.0, reasons, vault)
        else:
            decision = Decision("allow", text, findings, 0.0, reasons, vault)
        self._record("out", decision, text, dry_run)
        return decision

    # -- wrapping --------------------------------------------------------

    def wrap(self, fn, *, dry_run: bool = False):
        """Wrap a ``str -> str`` callable with input + output guards.

        Blocked input raises ``BlockedError`` before ``fn`` is called.
        Redacted input is restored on the way out: placeholders the
        model echoes back become the user's own values again.
        """
        guard = self

        def guarded(text: str) -> str:
            incoming = guard.check_input(text, dry_run=dry_run)
            if incoming.blocked:
                raise BlockedError(
                    "input blocked by guardkit: " + "; ".join(incoming.reasons),
                    incoming,
                )
            result = fn(incoming.safe_text)
            outgoing = guard.check_output(result, dry_run=dry_run)
            if outgoing.blocked:
                raise BlockedError(
                    "output blocked by guardkit: " + "; ".join(outgoing.reasons),
                    outgoing,
                )
            restored = incoming.vault.restore(outgoing.safe_text) if incoming.vault else outgoing.safe_text
            return restored

        return guarded

    # -- internals -------------------------------------------------------

    def _record(
        self, direction: str, decision: Decision, text: str, dry_run: bool
    ) -> None:
        if self.audit is not None and not dry_run:
            self.audit.record(
                direction=direction,
                action=decision.action,
                text=text,
                findings=decision.findings,
                score=decision.score,
                reasons=decision.reasons,
            )


class BlockedError(RuntimeError):
    """Raised by Guard.wrap when input or output is blocked."""

    def __init__(self, message: str, decision: Decision):
        super().__init__(message)
        self.decision = decision
