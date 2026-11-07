"""Model adapters for evalkit.

An adapter is anything with ``generate(prompt, seed=0) -> str``. Two ship
with evalkit:

* :class:`StubAdapter` — deterministic canned responses, driven by a
  fixtures file. Used for tests and demos; no network, no key.
* :class:`OpenAICompatibleAdapter` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint over plain ``urllib``. The API key comes
  from the environment only (``OPENAI_API_KEY`` by default, or the env var
  named by ``api_key_env``); it is never read from a file and never
  logged.

An optional LLM-as-judge is deliberately *not* built in: judging with a
model needs a key and makes runs non-deterministic, which defeats the
point of a regression harness. The :class:`ModelAdapter` interface is the
extension point — implement it and pass it to :func:`evalkit.run_suite`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


class AdapterError(Exception):
    """Raised when an adapter cannot produce a response."""


class ModelAdapter:
    """Minimal interface adapters must satisfy."""

    name: str = "base"

    def generate(self, prompt: str, seed: int = 0) -> str:
        raise NotImplementedError


class StubAdapter(ModelAdapter):
    """Deterministic canned responses, matched by prompt substring.

    Fixtures file shape::

        {
          "default": "I don't know.",
          "rules": [
            {"match": "quarterly earnings call", "response": "..."}
          ]
        }

    The first rule whose ``match`` string appears (case-insensitively) in
    the prompt wins; otherwise ``default`` is returned. Matching is
    substring-based on purpose: it keeps fixtures readable and makes the
    demo golden set self-explanatory.
    """

    name = "stub"

    def __init__(self, rules: list[dict] | None = None, default: str = ""):
        self.rules = rules or []
        self.default = default

    @classmethod
    def from_file(cls, path: str | Path) -> "StubAdapter":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"cannot load stub fixtures {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("rules", []), list):
            raise AdapterError(f"stub fixtures {path} need a 'rules' list")
        return cls(rules=data.get("rules", []), default=data.get("default", ""))

    def generate(self, prompt: str, seed: int = 0) -> str:
        lowered = prompt.lower()
        for rule in self.rules:
            match = rule.get("match", "")
            if match and match.lower() in lowered:
                return rule.get("response", "")
        return self.default


class OpenAICompatibleAdapter(ModelAdapter):
    """Minimal OpenAI-compatible chat-completions client over urllib.

    ``api_key`` defaults to the ``OPENAI_API_KEY`` environment variable
    (or ``api_key_env``). ``base_url`` defaults to OpenAI's API; point it
    at any compatible endpoint (local server, gateway, other provider).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 60,
    ):
        self.model = model or os.environ.get("EVALKIT_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise AdapterError(
                f"no API key: set the {api_key_env} environment variable "
                "(keys are never read from files)"
            )
        self.timeout = timeout
        self.name = f"openai-compatible:{self.model}"

    def generate(self, prompt: str, seed: int = 0) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "seed": seed,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AdapterError(f"model API returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise AdapterError(f"model API unreachable: {exc.reason}") from exc
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(f"unexpected model API response shape: {exc}") from exc


def build_adapter(kind: str, **kwargs) -> ModelAdapter:
    """Build an adapter by name: ``stub`` or ``openai``."""
    if kind == "stub":
        fixtures = kwargs.get("fixtures")
        if fixtures is None:
            raise AdapterError("stub adapter needs --fixtures <file>")
        return StubAdapter.from_file(fixtures)
    if kind == "openai":
        return OpenAICompatibleAdapter(
            model=kwargs.get("model"),
            base_url=kwargs.get("base_url"),
            api_key_env=kwargs.get("api_key_env", "OPENAI_API_KEY"),
        )
    raise AdapterError(f"unknown adapter {kind!r} (expected 'stub' or 'openai')")
