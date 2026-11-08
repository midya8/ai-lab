"""Model backends: a deterministic offline stub for demos/tests and a command adapter for real models."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess

LABELS = ("billing", "bug", "account", "feature")
_KEYWORDS = {
    "billing": ("charge", "invoice", "refund", "payment", "billed", "subscription fee"),
    "bug": ("crash", "error", "exception", "broken", "freezes", "not loading"),
    "account": ("password", "login", "log in", "locked out", "two-factor", "email address"),
    "feature": ("would love", "please add", "feature request", "wish", "support for"),
}


class RuleModel:
    """A stand-in for an LLM that is *sensitive to prompt wording*, deterministically.

    It classifies a support ticket by keyword, then makes a hash-seeded mistake
    with a probability that depends on prompt quality: listing the allowed
    categories, delimiting the ticket with ``<ticket>`` tags, and including an
    example each lower the error rate. Output format follows the prompt too
    (JSON / single word / chatty). The error rates are invented for the demo;
    they are not measurements of any real model.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _ticket(self, prompt: str) -> str:
        found = re.findall(r"<ticket>(.*?)</ticket>", prompt, re.S)  # last one: earlier ones are examples
        return (found[-1] if found else prompt.strip().splitlines()[-1]).strip()

    def _classify(self, text: str) -> str:
        low = text.lower()
        best, hits = "feature", 0
        for label, words in _KEYWORDS.items():
            h = sum(w in low for w in words)
            if h > hits:
                best, hits = label, h
        return best

    def __call__(self, prompt: str) -> str:
        label = self._classify(self._ticket(prompt))
        low = prompt.lower()
        p_err = 0.34
        if "categories" in low:
            p_err -= 0.14
        if "<ticket>" in low:
            p_err -= 0.08
        if "example" in low:
            p_err -= 0.06
        h = int(hashlib.sha256(f"{self.seed}|{prompt}".encode()).hexdigest()[:8], 16)
        if h / 0xFFFFFFFF < p_err:
            wrong = [l for l in LABELS if l != label]
            label = wrong[h % len(wrong)]
        if "json" in low:
            return json.dumps({"category": label})
        if "one word" in low or "single word" in low:
            return label
        if "concise" in low or "brief" in low:
            return f"This looks like a {label} issue."
        return (f"Thanks for sharing this ticket! After reading it carefully, my assessment is that this is "
                f"most likely a {label} issue, although there could be other factors involved.")


class CommandModel:
    """Run an external command per prompt (prompt on stdin, answer on stdout), no shell involved."""

    def __init__(self, command: str, timeout: float = 120.0):
        self.argv = shlex.split(command)
        if not self.argv:
            raise ValueError("empty model command")
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        r = subprocess.run(self.argv, input=prompt, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {r.stderr.strip()[:200]}")
        return r.stdout


def make_model(spec: str):
    """``rule`` | ``rule:<seed>`` | ``cmd:<command line>``."""
    if spec == "rule" or spec.startswith("rule:"):
        return RuleModel(int(spec.split(":", 1)[1]) if ":" in spec else 0)
    if spec.startswith("cmd:"):
        return CommandModel(spec[4:])
    raise ValueError(f"unknown model spec {spec!r} (use rule, rule:<seed> or cmd:<command>)")


class RuleJudge:
    """Offline stand-in for a judge LLM, for demos and tests (not a real grader).

    It reads the rubric's double-quoted terms: terms in a sentence containing
    avoid/never/must not/without are *forbidden*, all others *required*. The
    score is ``1 + 4 * share of required terms present`` (word-boundary,
    case-insensitive), minus 2 per forbidden term present, clamped to 1..5.
    It only measures lexical coverage, so calibrate it before believing it.
    """

    _NEG = re.compile(r"\b(avoid|never|must not|should not|without|do not|don't)\b", re.I)

    def __call__(self, prompt: str) -> str:
        rub = re.search(r"^<rubric>\n(.*?)\n</rubric>$", prompt, re.S | re.M)
        resp = re.search(r"^<response>\n(.*?)\n</response>$", prompt, re.S | re.M)
        rubric = rub.group(1) if rub else ""
        response = (resp.group(1) if resp else "").lower()
        required, forbidden = [], []
        for sentence in re.split(r"(?<=[.;!?])\s+|\n", rubric):
            terms = re.findall(r'"([^"]+)"', sentence)
            (forbidden if self._NEG.search(sentence) else required).extend(t.lower() for t in terms)

        def present(t: str) -> bool:
            return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", response) is not None

        got = [t for t in required if present(t)]
        bad = [t for t in forbidden if present(t)]
        cover = len(got) / len(required) if required else 1.0
        score = max(1, min(5, round(1 + 4 * cover) - 2 * len(bad)))
        reason = f"covers {len(got)}/{len(required)} required terms" + (f"; contains forbidden {bad}" if bad else "")
        return json.dumps({"score": score, "reason": reason})


def make_judge(spec: str | None):
    """``None`` | ``rule`` | ``cmd:<command>`` (the prompt goes on stdin, the reply is read from stdout)."""
    if spec is None or spec == "":
        return None
    if spec == "rule":
        return RuleJudge()
    if spec.startswith("cmd:"):
        return CommandModel(spec[4:])
    raise ValueError(f"unknown judge spec {spec!r} (use rule or cmd:<command>)")
