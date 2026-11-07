"""Pluggable model backends. Anything with generate(prompt)->str works."""
import json
import re
import urllib.request


class Backend:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class EchoBackend(Backend):
    """Returns the prompt's last line; a trivial baseline."""
    name = "echo"

    def generate(self, prompt):
        lines = prompt.strip().splitlines()
        return lines[-1] if lines else ""


class RuleBasedBackend(Backend):
    """Offline deterministic 'model': solves arithmetic and a few lookups.
    Lets the harness run in CI with no network or API keys."""
    name = "rules"
    FACTS = {"capital of france": "Paris", "capital of japan": "Tokyo",
             "largest planet": "Jupiter"}

    def generate(self, prompt):
        m = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", prompt)
        if m:
            a, op, b = int(m[1]), m[2], int(m[3])
            if op == "/" and b == 0:
                return "undefined"
            r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
            return str(int(r) if float(r).is_integer() else round(r, 4))
        low = prompt.lower()
        for k, v in self.FACTS.items():
            if k in low:
                return f"The answer is {v}."
        if "json" in low:
            return json.dumps({"ok": True})
        return "I don't know."


class HTTPBackend(Backend):
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, Ollama)."""

    def __init__(self, url, model, api_key="", timeout=60):
        self.url, self.model, self.api_key, self.timeout = url, model, api_key, timeout
        self.name = f"http:{model}"

    def generate(self, prompt):
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, body, {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


def get_backend(spec: str, **kw) -> Backend:
    if spec == "echo":
        return EchoBackend()
    if spec == "rules":
        return RuleBasedBackend()
    if spec == "http":
        return HTTPBackend(**kw)
    raise ValueError(f"unknown backend {spec!r}")
