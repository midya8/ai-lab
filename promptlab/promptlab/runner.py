"""Run a prompt version over a test set and record per-case outcomes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from .checks import run_check
from .registry import Version
from .template import Template, TemplateError

Model = Callable[[str], str]


@dataclass
class Case:
    id: str
    vars: dict
    checks: list[dict]


@dataclass
class CaseResult:
    id: str
    passed: bool
    score: float
    output: str
    failures: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class RunResult:
    prompt: str
    version: int
    hash: str
    results: list[CaseResult]
    prompt_words: float = 0.0
    output_words: float = 0.0

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / self.n if self.n else 0.0

    def by_id(self) -> dict[str, CaseResult]:
        return {r.id: r for r in self.results}

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunResult":
        return RunResult(d["prompt"], d["version"], d["hash"],
                         [CaseResult(**r) for r in d["results"]],
                         d.get("prompt_words", 0.0), d.get("output_words", 0.0))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "RunResult":
        return RunResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_cases(path: str | Path) -> list[Case]:
    """JSONL: one ``{"id", "vars", "checks"}`` object per line; ``#`` lines and blanks ignored."""
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
            case = Case(str(d["id"]), dict(d.get("vars", {})), list(d["checks"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}:{lineno}: bad case ({e!r})") from None
        if case.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        if not case.checks:
            raise ValueError(f"{path}:{lineno}: case {case.id!r} has no checks")
        seen.add(case.id)
        cases.append(case)
    return cases


def run_prompt(version: Version, cases: list[Case], model: Model) -> RunResult:
    """Render the version for each case, call ``model`` and score the output.

    A render error or a model exception fails that case (score 0) and is
    recorded; it never aborts the run.
    """
    tpl: Template = version.template
    results: list[CaseResult] = []
    pw = ow = 0
    for c in cases:
        try:
            prompt = tpl.render(c.vars)
        except TemplateError as e:
            results.append(CaseResult(c.id, False, 0.0, "", [], f"render: {e}"))
            continue
        pw += len(prompt.split())
        try:
            out = model(prompt)
        except Exception as e:  # noqa: BLE001 - model backends can fail any way
            results.append(CaseResult(c.id, False, 0.0, "", [], f"model: {e}"))
            continue
        ow += len(out.split())
        checks = [run_check(ch, out) for ch in c.checks]
        ok = sum(ch.passed for ch in checks)
        fails = [f"{ch.type}: {ch.detail}" if ch.detail else ch.type for ch in checks if not ch.passed]
        results.append(CaseResult(c.id, ok == len(checks), ok / len(checks), out, fails))
    n = max(len(results), 1)
    return RunResult(version.name, version.number, version.hash, results, pw / n, ow / n)
