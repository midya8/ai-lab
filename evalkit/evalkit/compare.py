"""Regression detection: diff two evalkit run files.

A case *regressed* when it passed in the baseline and fails in the new
run. A case *improved* when the reverse holds. Score deltas are reported
per case and averaged per tag, so a prompt change that helps one area
while hurting another shows up instead of cancelling out silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .runner import RunResult


@dataclass
class CompareResult:
    baseline_path: str
    run_path: str
    baseline_score: float
    run_score: float
    newly_failing: list[str] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    score_deltas: dict[str, float] = field(default_factory=dict)
    tag_deltas: dict[str, float] = field(default_factory=dict)
    min_score: float = 0.0

    @property
    def regressed(self) -> bool:
        """True when anything got worse or the run is below --min-score."""
        return bool(self.newly_failing) or self.run_score < self.min_score

    @property
    def overall_delta(self) -> float:
        return round(self.run_score - self.baseline_score, 4)


def compare_runs(
    baseline: RunResult, run: RunResult, min_score: float = 0.0
) -> CompareResult:
    base = {c.case_id: c for c in baseline.cases}
    new = {c.case_id: c for c in run.cases}

    newly_failing = [cid for cid, c in new.items() if cid in base and base[cid].passed and not c.passed]
    newly_passing = [cid for cid, c in new.items() if cid in base and not base[cid].passed and c.passed]
    added = [cid for cid in new if cid not in base]
    removed = [cid for cid in base if cid not in new]
    deltas = {
        cid: round(new[cid].score - base[cid].score, 4)
        for cid in new
        if cid in base and new[cid].score != base[cid].score
    }

    # Per-tag mean delta over cases present in both runs.
    tag_scores: dict[str, list[float]] = {}
    for cid in new:
        if cid not in base:
            continue
        delta = new[cid].score - base[cid].score
        for tag in new[cid].tags:
            tag_scores.setdefault(tag, []).append(delta)
    tag_deltas = {
        tag: round(sum(v) / len(v), 4) for tag, v in sorted(tag_scores.items())
    }

    return CompareResult(
        baseline_path=getattr(baseline, "_path", ""),
        run_path=getattr(run, "_path", ""),
        baseline_score=round(baseline.overall_score, 4),
        run_score=round(run.overall_score, 4),
        newly_failing=sorted(newly_failing),
        newly_passing=sorted(newly_passing),
        added=sorted(added),
        removed=sorted(removed),
        score_deltas=deltas,
        tag_deltas=tag_deltas,
        min_score=min_score,
    )


def format_compare(result: CompareResult) -> str:
    """Human-readable summary of a comparison (also printed by the CLI)."""
    lines = [
        f"baseline overall score: {result.baseline_score:.3f}",
        f"run      overall score: {result.run_score:.3f} "
        f"(delta {result.overall_delta:+.3f})",
        f"min-score threshold:    {result.min_score:.3f}",
        "",
    ]
    if result.newly_failing:
        lines.append(f"newly failing ({len(result.newly_failing)}):")
        lines.extend(f"  - {cid}" for cid in result.newly_failing)
    else:
        lines.append("newly failing: none")
    if result.newly_passing:
        lines.append(f"newly passing ({len(result.newly_passing)}):")
        lines.extend(f"  + {cid}" for cid in result.newly_passing)
    else:
        lines.append("newly passing: none")
    if result.added:
        lines.append(f"added cases: {', '.join(result.added)}")
    if result.removed:
        lines.append(f"removed cases: {', '.join(result.removed)}")
    if result.tag_deltas:
        lines.append("")
        lines.append("score delta by tag:")
        for tag, delta in result.tag_deltas.items():
            lines.append(f"  {tag}: {delta:+.3f}")
    if result.score_deltas:
        lines.append("")
        lines.append("per-case score deltas:")
        for cid, delta in sorted(result.score_deltas.items()):
            lines.append(f"  {cid}: {delta:+.3f}")
    lines.append("")
    lines.append("REGRESSION" if result.regressed else "no regression")
    return "\n".join(lines)
