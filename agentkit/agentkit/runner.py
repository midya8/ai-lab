"""The agent loop for agentkit.

One run: plan -> call tools -> observe results -> decide the next
step, until the provider answers in text, the step limit is hit, the
same tool call repeats, or a tool times out. Guardrails:

* only tools in the task's allowlist may run;
* tool-call arguments are size-capped;
* each tool call has a per-step timeout;
* unknown tool calls are rejected, never executed;
* no shell execution unless the shell tool was explicitly registered.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from .model import (
    RunResult,
    StepRecord,
    Task,
    ToolCall,
    ValidationError,
)
from .tools import ToolRegistry

DEFAULT_MAX_ARGS_BYTES = 4096


class AgentRunner:
    """Runs a task against a provider and a tool registry."""

    def __init__(
        self,
        provider,
        registry: ToolRegistry,
        max_steps: int = 10,
        timeout_seconds: float = 30.0,
        max_args_bytes: int = DEFAULT_MAX_ARGS_BYTES,
    ) -> None:
        if max_steps < 1:
            raise ValidationError("max_steps must be >= 1")
        self.provider = provider
        self.registry = registry
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.max_args_bytes = max_args_bytes

    # -- planning ------------------------------------------------------

    def plan(self, task: Task) -> dict:
        """Return the run plan without executing anything (dry run)."""
        allowed = self._allowed_tools(task)
        return {
            "task": task.text,
            "provider": type(self.provider).__name__,
            "max_steps": min(task.max_steps, self.max_steps),
            "timeout_seconds": task.timeout_seconds,
            "tools": [t.schema() for t in allowed],
        }

    def _allowed_tools(self, task: Task) -> list:
        wanted = task.tools if task.tools is not None else self.registry.names()
        allowed = []
        for name in wanted:
            tool = self.registry.get(name)
            if tool is None:
                raise ValidationError(f"task requests unknown tool: {name!r}")
            allowed.append(tool)
        return allowed

    # -- running -------------------------------------------------------

    def run(self, task: Task, dry_run: bool = False) -> RunResult:
        """Run the task and return a RunResult (never raises for agent
        failures; invalid input raises ValidationError)."""
        started = time.monotonic()
        allowed = self._allowed_tools(task)
        allowed_names = {t.name for t in allowed}
        schemas = [t.schema() for t in allowed]
        limit = min(task.max_steps, self.max_steps)
        steps: list[StepRecord] = []
        messages = [{"role": "user", "content": task.text}]
        usage: dict = {}
        status = "success"
        final_answer = None

        if dry_run:
            return RunResult(
                status="dry_run",
                task=task.text,
                steps=[],
                duration_seconds=0.0,
                usage={"planned_steps": limit, "tools": sorted(allowed_names)},
            )

        seen_calls: set[str] = set()
        for step_no in range(1, limit + 1):
            step_started = time.monotonic()
            record = StepRecord(step=step_no)
            try:
                response = self.provider.complete(messages, schemas)
            except Exception as exc:  # provider failure ends the run
                record.error = f"provider error: {exc}"
                record.duration_seconds = time.monotonic() - step_started
                steps.append(record)
                status = "failed"
                break
            usage.update(response.usage or {})
            if response.text is not None:
                record.provider_text = response.text
                record.duration_seconds = time.monotonic() - step_started
                steps.append(record)
                final_answer = response.text
                status = "success"
                break
            call = response.tool_call
            record.tool_call = call
            error = self._check_call(call, allowed_names, seen_calls)
            if error is not None:
                record.error = error
                record.duration_seconds = time.monotonic() - step_started
                steps.append(record)
                status = "loop_detected" if "repeated" in error else "failed"
                break
            seen_calls.add(call.canonical())
            result, tool_error = self._execute(call, task.timeout_seconds)
            record.duration_seconds = time.monotonic() - step_started
            if tool_error is not None:
                record.error = tool_error
                steps.append(record)
                status = "timeout" if "timed out" in tool_error else "failed"
                break
            record.tool_result = result
            steps.append(record)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Called {call.name} with {json.dumps(call.arguments)}",
                }
            )
            messages.append({"role": "tool", "content": result})
        else:
            # Loop exhausted without a final answer.
            status = "step_limit"

        return RunResult(
            status=status,
            task=task.text,
            steps=steps,
            final_answer=final_answer,
            duration_seconds=time.monotonic() - started,
            usage=usage,
        )

    def _check_call(
        self, call: ToolCall, allowed_names: set, seen_calls: set
    ) -> str | None:
        """Guardrail checks before a tool runs. Returns an error string
        or None when the call may proceed."""
        if call.name not in allowed_names:
            return f"unknown tool rejected: {call.name!r}"
        if not isinstance(call.arguments, dict):
            return f"tool {call.name!r}: arguments must be an object"
        try:
            encoded = json.dumps(call.arguments).encode("utf-8")
        except (TypeError, ValueError):
            return f"tool {call.name!r}: arguments are not JSON-serializable"
        if len(encoded) > self.max_args_bytes:
            return (
                f"tool {call.name!r}: arguments exceed "
                f"{self.max_args_bytes} bytes ({len(encoded)} bytes)"
            )
        if call.canonical() in seen_calls:
            return (
                f"repeated identical tool call rejected: "
                f"{call.name} {json.dumps(call.arguments, sort_keys=True)}"
            )
        return None

    def _execute(self, call: ToolCall, timeout: float) -> tuple[str | None, str | None]:
        """Execute one validated tool call with a per-step timeout.

        Returns (result, error); exactly one is None.
        """
        tool = self.registry.get(call.name)
        if tool is None:  # cannot happen after _check_call, defensive
            return None, f"unknown tool rejected: {call.name!r}"
        try:
            tool.validate_args(call.arguments)
        except ValidationError as exc:
            return None, str(exc)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.func, **call.arguments)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeout:
            # Do not wait for the hung thread: abandon it and fail fast.
            pool.shutdown(wait=False, cancel_futures=True)
            return None, f"tool {call.name!r} timed out after {timeout}s"
        except Exception as exc:
            pool.shutdown(wait=True)
            return None, f"tool {call.name!r} failed: {exc}"
        pool.shutdown(wait=True)
        return (result if isinstance(result, str) else str(result)), None
