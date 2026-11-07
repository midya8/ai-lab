"""guardkit — runtime guardrails for LLM pipelines.

Detectors, redaction, a policy engine, an audit log, and an adversarial
eval harness. Standard library only, fully offline.

This is *runtime* guardrails: it inspects live prompts and live model
output as they flow through a pipeline, and decides allow / redact /
block. It is the complement of promptguard (a separate standalone repo),
which lints prompt *files* statically for CI. guardkit does not scan
files; it guards requests.
"""

from .audit import AuditLog
from .detectors import Finding, InjectionScorer, PiiSecretScanner, scan_text
from .eval import EvalReport, evaluate_cases, load_cases
from .guard import Decision, Guard
from .policy import Policy, PolicyError, default_policy, load_policy
from .redact import RedactionVault, redact_spans

__all__ = [
    "AuditLog",
    "Decision",
    "EvalReport",
    "Finding",
    "Guard",
    "InjectionScorer",
    "PiiSecretScanner",
    "Policy",
    "PolicyError",
    "RedactionVault",
    "default_policy",
    "evaluate_cases",
    "load_cases",
    "load_policy",
    "redact_spans",
    "scan_text",
]

__version__ = "0.1.0"
