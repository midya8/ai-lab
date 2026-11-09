"""Run a prompt version over a test set and record per-case outcomes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from .checks import run_check
from .registry import Version
from .cost import BudgetExceeded
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
    samples: int = 1
    pass_count: int = -1  # -1 = legacy file; derived from ``passed``

    def __post_init__(self):
        if self.pass_count < 0:
            self.pass_count = self.samples if self.passed else 0

    @property
    def flaky(self) -> bool:
        """Some, but not all, samples passed."""
        return 0 < self.pass_count < self.samples


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

    @property
    def flaky(self) -> list[str]:
        return [r.id for r in self.results if r.flaky]

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


def run_prompt(version: Version, cases: list[Case], model: Model,
               samples: int = 1, threshold: float = 1.0, judge: Model | None = None) -> RunResult:
    """Render the version for each case, call ``model`` and score the output.

    With ``samples`` > 1 the model is called that many times per case (for
    non-deterministic models). A case passes when at least ``threshold`` of
    its samples pass every check (default 1.0: all of them); its score is the
    mean over samples. Cases with mixed outcomes are reported as flaky.
    The stored output is the first failing sample's, else the first.

    ``judge`` is the model used by ``judge`` checks (without one they fail).

    A render error or a model exception fails that sample (score 0) and is
    recorded; it never aborts the run.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    tpl: Template = version.template
    results: list[CaseResult] = []
    pw = ow = calls = 0
    for c in cases:
        try:
            prompt = tpl.render(c.vars)
        except TemplateError as e:
            results.append(CaseResult(c.id, False, 0.0, "", [], f"render: {e}", samples, 0))
            continue
        scores: list[float] = []
        wins = 0
        out_keep, fails_keep, err_keep = None, [], ""
        for _ in range(samples):
            pw += len(prompt.split())
            try:
                out = model(prompt)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - model backends can fail any way
                scores.append(0.0)
                if out_keep is None or not err_keep and not fails_keep:
                    out_keep, err_keep = "", f"model: {e}"
                continue
            calls += 1
            ow += len(out.split())
            checks = [run_check(ch, out, judge, c.vars) for ch in c.checks]
            ok = sum(ch.passed for ch in checks)
            scores.append(ok / len(checks))
            if ok == len(checks):
                wins += 1
                if out_keep is None:
                    out_keep = out
            else:
                fails = [f"{ch.type}: {ch.detail}" if ch.detail else ch.type for ch in checks if not ch.passed]
                if not fails_keep and not err_keep:
                    out_keep, fails_keep = out, fails
        passed = wins / samples >= threshold - 1e-12
        results.append(CaseResult(c.id, passed, sum(scores) / samples, out_keep or "",
                                  fails_keep, err_keep, samples, wins))
    n = max(len(results), 1)
    return RunResult(version.name, version.number, version.hash, results,
                     pw / (n * samples), ow / max(calls, 1))
