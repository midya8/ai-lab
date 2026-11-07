"""Running a golden set against an adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .adapters import AdapterError, ModelAdapter
from .cases import Case
from .scorers import CaseScore, score_response


@dataclass
class CaseRun:
    case_id: str
    tags: list[str]
    score: float
    passed: bool
    checks: list[dict]
    response: str
    flaky: bool = False
    repeats: int = 1


@dataclass
class RunResult:
    adapter: str
    seed: int
    repeat: int
    cases: list[CaseRun] = field(default_factory=list)
    created_at: str = ""

    @property
    def overall_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed_count(self) -> int:
        return len(self.cases) - self.passed_count

    @property
    def flaky_ids(self) -> list[str]:
        return [c.case_id for c in self.cases if c.flaky]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["overall_score"] = round(self.overall_score, 4)
        data["passed_count"] = self.passed_count
        data["failed_count"] = self.failed_count
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        cases = [CaseRun(**c) for c in data.get("cases", [])]
        return cls(
            adapter=data.get("adapter", ""),
            seed=data.get("seed", 0),
            repeat=data.get("repeat", 1),
            cases=cases,
            created_at=data.get("created_at", ""),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load run file {path}: {exc}") from exc
        if not isinstance(data, dict) or "cases" not in data:
            raise ValueError(f"run file {path} is not an evalkit run result")
        return cls.from_dict(data)


def run_suite(
    cases: list[Case],
    adapter: ModelAdapter,
    repeat: int = 1,
    seed: int = 0,
) -> RunResult:
    """Run every case against the adapter.

    With ``repeat > 1`` each prompt is generated that many times. A case
    is flagged ``flaky`` when its score differs across repeats — a
    determinism failure worth investigating even if the first run passed.
    The reported score is the first repeat's score; repeats only affect
    the flaky flag.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    results: list[CaseRun] = []
    for case in cases:
        scores: list[CaseScore] = []
        responses: list[str] = []
        for i in range(repeat):
            try:
                response = adapter.generate(case.prompt, seed=seed + i)
            except AdapterError:
                raise
            except Exception as exc:  # adapter bugs surface as exit code 2
                raise AdapterError(
                    f"adapter {adapter.name!r} failed on case {case.id!r}: {exc}"
                ) from exc
            responses.append(response)
            scores.append(score_response(case, response))
        first = scores[0]
        flaky = any(s.score != first.score for s in scores[1:])
        results.append(
            CaseRun(
                case_id=case.id,
                tags=list(case.tags),
                score=first.score,
                passed=first.passed,
                checks=[{"name": c.name, "passed": c.passed, "detail": c.detail} for c in first.checks],
                response=responses[0],
                flaky=flaky,
                repeats=repeat,
            )
        )
    return RunResult(
        adapter=adapter.name,
        seed=seed,
        repeat=repeat,
        cases=results,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
