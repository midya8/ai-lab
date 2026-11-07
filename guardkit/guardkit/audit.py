"""Append-only JSONL audit log for guard decisions.

Every entry records the decision, the score, the rule ids that fired,
and the *redacted* text — never a raw secret, email, card number, or
phone number. The vault mapping is never written here: if the process
exits, the redacted log stays reviewable and the values stay gone.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .detectors import Finding
from .redact import redact_spans

__all__ = ["AuditLog"]


class AuditLog:
    """Appends one JSON object per guard decision."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(
        self,
        *,
        direction: str,
        action: str,
        text: str,
        findings: list[Finding],
        score: float = 0.0,
        reasons: list[str] | None = None,
    ) -> dict:
        redacted, _ = redact_spans(text, findings)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "direction": direction,
            "action": action,
            "score": round(score, 3),
            "rules": sorted({f.rule_id for f in findings}),
            "reasons": reasons or [],
            "text": redacted,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
