"""LLM providers for agentkit.

A provider is anything with one method::

    complete(messages, tools) -> ProviderResponse

``messages`` is a list of ``{"role": ..., "content": ...}`` dicts and
``tools`` is a list of JSON Schemas (see :class:`agentkit.model.Tool`).
Two providers ship:

* :class:`StubProvider` — deterministic, rule-based, no network. Used
  for tests, demos, and CI.
* :class:`OpenAICompatibleProvider` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint over ``urllib``. The API key comes
  from the ``AGENTKIT_API_KEY`` environment variable only, and only
  when the user opts in by choosing this provider. It is never called
  by the test suite.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .model import AgentError, ToolCall

_ADD_RE = re.compile(
    r"\badd\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", re.IGNORECASE
)
_SUM_RE = re.compile(
    r"\bsum\s+of\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", re.IGNORECASE
)
_ARITH_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)"
)
_SEARCH_RE = re.compile(
    r"\b(?:search|find)\b.*?\b(?:for|about)\s+['\"]?([^'\"\n]+?)['\"]?\s*(?:in\b|$)|"
    r"\b(?:search|find)\b.*?\bnotes?\s+['\"]?([^'\"\n]+?)['\"]?\s*$",
    re.IGNORECASE,
)
_READ_RE = re.compile(r"\bread\s+(?:the\s+file\s+)?['\"]?([A-Za-z0-9_./-]+\.[\w]+)['\"]?", re.IGNORECASE)


@dataclass
class ProviderResponse:
    """What a provider returns for one step.

    Exactly one of ``tool_call`` and ``text`` is set: a tool call means
    "run this and show me the result", text means "I am done, this is
    the final answer".
    """

    tool_call: ToolCall | None = None
    text: str | None = None
    usage: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.tool_call is None) == (self.text is None):
            raise ValueError(
                "ProviderResponse needs exactly one of tool_call or text"
            )


class Provider(Protocol):
    """The interface every provider implements."""

    def complete(self, messages: list, tools: list) -> ProviderResponse:
        ...


class StubProvider:
    """A deterministic, rule-based provider. No network, no API key.

    It understands three intents in the task text, in this order:
    arithmetic (``add 17 and 25``, ``17 + 25``), reading a file
    (``read notes.md``), and searching notes (``search notes for X``).
    After a tool result arrives it answers with the result. Anything
    else gets a plain-text answer saying no tool applies.

    A ``script`` may be supplied instead: a list of ProviderResponses
    returned in order, which tests use to drive exact loop/step-limit
    behaviour. When the script runs out, the provider answers in text.
    The provider is fully deterministic: the same messages always
    produce the same response.
    """

    def __init__(self, script: list | None = None) -> None:
        self._script = list(script) if script else []
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def complete(self, messages: list, tools: list) -> ProviderResponse:
        self._calls += 1
        if self._script:
            return self._script.pop(0)
        available = {t["name"] for t in tools}
        task = messages[0]["content"] if messages else ""
        last_tool = next(
            (m for m in reversed(messages) if m.get("role") == "tool"), None
        )
        if last_tool is not None:
            # A tool just ran: answer with its result.
            return ProviderResponse(
                text=f"Done. Result: {last_tool['content']}",
                usage={"stub_calls": self._calls},
            )
        # First step: pick a tool from the task text.
        match = _ADD_RE.search(task) or _SUM_RE.search(task)
        if match and "calculator" in available:
            expr = f"{match.group(1)} + {match.group(2)}"
            return ProviderResponse(
                tool_call=ToolCall(name="calculator", arguments={"expression": expr}),
                usage={"stub_calls": self._calls},
            )
        match = _ARITH_RE.search(task)
        if match and "calculator" in available:
            expr = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            return ProviderResponse(
                tool_call=ToolCall(name="calculator", arguments={"expression": expr}),
                usage={"stub_calls": self._calls},
            )
        match = _READ_RE.search(task)
        if match and "read_file" in available:
            return ProviderResponse(
                tool_call=ToolCall(
                    name="read_file", arguments={"path": match.group(1)}
                ),
                usage={"stub_calls": self._calls},
            )
        match = _SEARCH_RE.search(task)
        if match and "search_notes" in available:
            query = (match.group(1) or match.group(2) or "").strip().strip("'\"")
            return ProviderResponse(
                tool_call=ToolCall(
                    name="search_notes", arguments={"query": query}
                ),
                usage={"stub_calls": self._calls},
            )
        return ProviderResponse(
            text=(
                "I could not map this task to an available tool, so I did "
                "not call one."
            ),
            usage={"stub_calls": self._calls},
        )


class OpenAICompatibleProvider:
    """An OpenAI-compatible HTTP provider using only ``urllib``.

    Reads its API key from ``AGENTKIT_API_KEY`` (or the ``api_key``
    argument) and its model from ``AGENTKIT_MODEL`` (default
    ``gpt-4o-mini``). The key is never logged and never stored. Only
    constructed when the user explicitly picks this provider; the test
    suite never instantiates it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("AGENTKIT_API_KEY")
        if not key:
            raise AgentError(
                "OpenAI-compatible provider needs AGENTKIT_API_KEY "
                "in the environment (no key is ever hardcoded)"
            )
        self._api_key = key
        self._base_url = (base_url or os.environ.get("AGENTKIT_BASE_URL")
                          or "https://api.openai.com/v1").rstrip("/")
        self._model = model or os.environ.get("AGENTKIT_MODEL") or "gpt-4o-mini"
        self._timeout = timeout_seconds

    def complete(self, messages: list, tools: list) -> ProviderResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": [
                {"type": "function", "function": t} for t in tools
            ],
            "tool_choice": "auto",
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AgentError(f"provider request failed ({exc.code}): {detail}")
        except urllib.error.URLError as exc:
            raise AgentError(f"provider request failed: {exc.reason}")
        message = body["choices"][0]["message"]
        usage = body.get("usage", {}) or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            first = tool_calls[0]["function"]
            try:
                args = json.loads(first.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return ProviderResponse(
                tool_call=ToolCall(name=first["name"], arguments=args),
                usage=usage,
            )
        return ProviderResponse(text=message.get("content") or "", usage=usage)
