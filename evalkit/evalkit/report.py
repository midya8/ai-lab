"""Markdown and JSON reports for an evalkit run.

Reports sort cases worst-first so the failures a human needs to look at
are at the top, not buried under a wall of green.
"""

from __future__ import annotations

import json
from pathlib import Path

from .runner import RunResult


def render_markdown(result: RunResult) -> str:
    lines = [
        "# evalkit report",
        "",
        f"- Adapter: `{result.adapter}`",
        f"- Seed: `{result.seed}` (repeat: `{result.repeat}`)",
        f"- Ran at: {result.created_at or 'unknown'}",
        f"- Overall score: **{result.overall_score:.3f}** "
        f"({result.passed_count} passed, {result.failed_count} failed, "
        f"{len(result.cases)} total)",
    ]
    if result.flaky_ids:
        lines.append(f"- Flaky cases (score changed across repeats): {', '.join(result.flaky_ids)}")
    lines += ["", "## Cases (worst first)", "", "| Case | Score | Result | Failed checks |", "|---|---|---|---|"]
    for case in sorted(result.cases, key=lambda c: (c.score, c.case_id)):
        failed = "; ".join(
            f"{c['name']} — {c['detail']}" for c in case.checks if not c["passed"]
        ) or "—"
        verdict = "pass" if case.passed else "FAIL"
        if case.flaky:
            verdict += " (flaky)"
        lines.append(f"| {case.case_id} | {case.score:.2f} | {verdict} | {failed} |")
    lines += ["", "## Responses", ""]
    for case in sorted(result.cases, key=lambda c: (c.score, c.case_id)):
        lines.append(f"### {case.case_id} (score {case.score:.2f})")
        lines.append("")
        lines.append("```")
        lines.append(case.response)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_json(result: RunResult) -> str:
    return json.dumps(result.to_dict(), indent=2)


def write_report(
    result: RunResult,
    md_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> list[str]:
    """Write the requested report files; returns the paths written."""
    written: list[str] = []
    if md_path is not None:
        path = Path(md_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(result), encoding="utf-8")
        written.append(str(path))
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_json(result), encoding="utf-8")
        written.append(str(path))
    return written
