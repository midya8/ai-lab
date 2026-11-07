"""Run traces and human-readable reports for agentkit.

Every run writes two artefacts: a JSON trace (machine-readable, one
record per step) and a short text report (readable at a glance).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .model import RunResult, Task


def build_trace(result: RunResult, task: Task, provider_name: str) -> dict:
    """The full JSON trace for a run."""
    return {
        "tool": "agentkit",
        "task": task.text,
        "provider": provider_name,
        "max_steps": task.max_steps,
        "timeout_seconds": task.timeout_seconds,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(result.duration_seconds, 3),
        "status": result.status,
        "final_answer": result.final_answer,
        "usage": result.usage,
        "steps": [step.to_dict() for step in result.steps],
    }


def write_trace(
    result: RunResult, task: Task, provider_name: str, path: str
) -> None:
    """Write the JSON trace to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_trace(result, task, provider_name), fh, indent=2)
        fh.write("\n")


def render_text_report(result: RunResult, task: Task, provider_name: str) -> str:
    """Render the human-readable report for a run."""
    lines = [
        "agentkit run report",
        f"task: {task.text}",
        f"provider: {provider_name}",
        f"status: {result.status}",
        f"steps: {len(result.steps)} of max {task.max_steps}",
        f"duration: {result.duration_seconds:.2f}s",
    ]
    if result.usage:
        lines.append(f"usage: {json.dumps(result.usage, sort_keys=True)}")
    for step in result.steps:
        lines.append(f"")
        lines.append(f"step {step.step}:")
        if step.provider_text is not None:
            lines.append(f"  answer: {step.provider_text}")
        if step.tool_call is not None:
            lines.append(
                f"  tool: {step.tool_call.name} "
                f"{json.dumps(step.tool_call.arguments, sort_keys=True)}"
            )
        if step.tool_result is not None:
            lines.append(f"  result: {step.tool_result}")
        if step.error is not None:
            lines.append(f"  error: {step.error}")
        lines.append(f"  took: {step.duration_seconds:.2f}s")
    if result.final_answer is not None:
        lines.append("")
        lines.append(f"final answer: {result.final_answer}")
    return "\n".join(lines) + "\n"


def write_text_report(
    result: RunResult, task: Task, provider_name: str, path: str
) -> None:
    """Write the text report to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_text_report(result, task, provider_name))
