"""reviewkit: a code-review bot for pull requests.

Deterministic rules (AST + regex) catch what can be caught reliably; an optional
LLM pass adds judgement, and every LLM comment is validated against the diff
before it can be posted.
"""

__version__ = "0.1.0"
