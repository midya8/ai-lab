"""Prompt templates with ``{{variable}}`` and ``{{variable|default}}`` placeholders."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\|([^{}]*))?\}\}")


class TemplateError(ValueError):
    """Raised when a template cannot be rendered with the given variables."""


@dataclass(frozen=True)
class Template:
    text: str

    def variables(self) -> list[str]:
        """Variable names in order of first appearance."""
        seen: list[str] = []
        for m in _PLACEHOLDER.finditer(self.text):
            if m.group(1) not in seen:
                seen.append(m.group(1))
        return seen

    def required(self) -> list[str]:
        """Variables that have no default at *any* of their occurrences."""
        with_default = {m.group(1) for m in _PLACEHOLDER.finditer(self.text) if m.group(2) is not None}
        return [v for v in self.variables() if v not in with_default]

    def render(self, values: dict[str, object] | None = None, *, strict: bool = True) -> str:
        """Substitute variables. Missing required variables raise unless ``strict=False``.

        Unknown extra values are rejected in strict mode, because a typo in a
        test case's variable name would otherwise silently render a bad prompt.
        """
        values = values or {}
        missing = [v for v in self.required() if v not in values]
        if missing and strict:
            raise TemplateError(f"missing variables: {', '.join(missing)}")
        extra = [k for k in values if k not in self.variables()]
        if extra and strict:
            raise TemplateError(f"unknown variables: {', '.join(sorted(extra))}")

        def sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            if name in values:
                return str(values[name])
            if default is not None:
                return default.strip()
            return m.group(0)  # non-strict: leave placeholder visible

        return _PLACEHOLDER.sub(sub, self.text)

    def digest(self) -> str:
        """Stable content hash (whitespace at line ends and file end ignored)."""
        norm = "\n".join(line.rstrip() for line in self.text.strip().splitlines())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()
