"""Policy file: what guardkit does when a detector fires.

A policy is a plain JSON file, so it can be reviewed, diffed, and
versioned like any other config. Unknown keys and out-of-range values
are rejected — a guardrail that silently ignores a typo is worse than
one that refuses to load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["Policy", "PolicyError", "default_policy", "load_policy"]

ACTIONS = ("allow", "redact", "block")
_DETECTOR_IDS = (
    "email",
    "phone",
    "card",
    "jwt",
    "private_key",
    "assigned_secret",
    "injection",
)

_DEFAULTS: dict = {
    "injection_threshold": 0.5,
    "injection_action": "block",
    "secret_action": "block",
    "pii_action": "redact",
    "max_input_chars": 20000,
    "enabled_detectors": list(_DETECTOR_IDS),
}


class PolicyError(ValueError):
    """Raised when a policy file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class Policy:
    """Decisions guardkit makes, as data."""

    injection_threshold: float = 0.5
    injection_action: str = "block"
    secret_action: str = "block"
    pii_action: str = "redact"
    max_input_chars: int = 20000
    enabled_detectors: tuple[str, ...] = _DETECTOR_IDS

    def validate(self) -> None:
        if not 0.0 < self.injection_threshold <= 1.0:
            raise PolicyError(
                f"injection_threshold must be in (0, 1], got {self.injection_threshold!r}"
            )
        for name in ("injection_action", "secret_action", "pii_action"):
            if getattr(self, name) not in ACTIONS:
                raise PolicyError(
                    f"{name} must be one of {ACTIONS}, got {getattr(self, name)!r}"
                )
        if self.injection_action == "redact":
            raise PolicyError(
                "injection_action cannot be 'redact': injection language "
                "has no safe span to redact, use 'block' or 'allow'"
            )
        if not isinstance(self.max_input_chars, int) or self.max_input_chars <= 0:
            raise PolicyError(
                f"max_input_chars must be a positive integer, got {self.max_input_chars!r}"
            )
        unknown = sorted(set(self.enabled_detectors) - set(_DETECTOR_IDS))
        if unknown:
            raise PolicyError(f"unknown detectors: {unknown}")
        if not self.enabled_detectors:
            raise PolicyError("enabled_detectors must not be empty")

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        if not isinstance(data, dict):
            raise PolicyError("policy must be a JSON object")
        unknown = sorted(set(data) - set(_DEFAULTS))
        if unknown:
            raise PolicyError(f"unknown policy keys: {unknown}")
        merged = {**_DEFAULTS, **data}
        detectors = merged["enabled_detectors"]
        if isinstance(detectors, list):
            merged["enabled_detectors"] = tuple(detectors)
        policy = cls(**merged)
        policy.validate()
        return policy

    def to_dict(self) -> dict:
        return {
            "injection_threshold": self.injection_threshold,
            "injection_action": self.injection_action,
            "secret_action": self.secret_action,
            "pii_action": self.pii_action,
            "max_input_chars": self.max_input_chars,
            "enabled_detectors": list(self.enabled_detectors),
        }


def default_policy() -> Policy:
    """The built-in defaults: block injection and secrets, redact PII."""
    return Policy.from_dict({})


def load_policy(path: str) -> Policy:
    """Load and validate a policy JSON file."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise PolicyError(f"policy file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise PolicyError(f"policy file is not valid JSON: {exc}") from None
    return Policy.from_dict(data)
