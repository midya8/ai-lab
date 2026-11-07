"""Command-line interface for agentkit.

Usage:
    python -m agentkit run examples/add-numbers.json
    python -m agentkit run task.json --provider openai --trace trace.json
    python -m agentkit run task.json --dry-run
    python -m agentkit validate task.json

Exit codes: 0 success (or a dry run / valid task), 1 agent failed,
step limit hit, loop detected, or a tool timed out, 2 invalid
task or tool definitions.
"""

from __future__ import annotations

import argparse
import json
import sys

from .model import Task, ValidationError
from .providers import OpenAICompatibleProvider, StubProvider
from .report import render_text_report, write_text_report, write_trace
from .runner import AgentRunner
from .tools import builtin_tools

EXIT_OK = 0
EXIT_AGENT_FAILED = 1
EXIT_INVALID = 2


def _build_provider(name: str):
    if name == "stub":
        return StubProvider()
    if name == "openai":
        return OpenAICompatibleProvider()
    raise ValidationError(f"unknown provider: {name!r} (use 'stub' or 'openai')")


def _load_task(path: str) -> Task:
    return Task.from_file(path)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        task = _load_task(args.task_file)
        provider = _build_provider(args.provider)
    except ValidationError as exc:
        print(f"agentkit: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except Exception as exc:  # provider construction (e.g. missing API key)
        print(f"agentkit: {exc}", file=sys.stderr)
        return EXIT_INVALID

    registry = builtin_tools(
        notes_dir=args.notes_dir or task.notes_dir,
        base_dir=args.base_dir,
        allow_shell=args.allow_shell,
    )
    try:
        runner = AgentRunner(
            provider,
            registry,
            max_steps=args.max_steps
            if args.max_steps is not None
            else task.max_steps,
            timeout_seconds=task.timeout_seconds,
        )
    except ValidationError as exc:
        print(f"agentkit: {exc}", file=sys.stderr)
        return EXIT_INVALID

    if args.dry_run:
        try:
            plan = runner.plan(task)
        except ValidationError as exc:
            print(f"agentkit: {exc}", file=sys.stderr)
            return EXIT_INVALID
        print("agentkit dry run — nothing will execute")
        print(f"  task: {plan['task']}")
        print(f"  provider: {plan['provider']}")
        print(f"  max steps: {plan['max_steps']}")
        print(f"  step timeout: {plan['timeout_seconds']}s")
        print(f"  tools: {', '.join(t['name'] for t in plan['tools'])}")
        for tool in plan["tools"]:
            print(f"    - {tool['name']}: {tool['description']}")
        return EXIT_OK

    try:
        result = runner.run(task)
    except ValidationError as exc:
        print(f"agentkit: {exc}", file=sys.stderr)
        return EXIT_INVALID

    provider_name = type(provider).__name__
    if args.trace:
        write_trace(result, task, provider_name, args.trace)
        print(f"trace written to {args.trace}")
    if args.report:
        write_text_report(result, task, provider_name, args.report)
        print(f"report written to {args.report}")
    print(render_text_report(result, task, provider_name))
    if result.status == "success":
        return EXIT_OK
    if result.status == "dry_run":
        return EXIT_OK
    return EXIT_AGENT_FAILED


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        task = _load_task(args.task_file)
    except ValidationError as exc:
        print(f"agentkit: invalid task: {exc}", file=sys.stderr)
        return EXIT_INVALID
    registry = builtin_tools(
        notes_dir=args.notes_dir or task.notes_dir,
        base_dir=args.base_dir,
        allow_shell=args.allow_shell,
    )
    try:
        registry.validate_all()
        runner = AgentRunner(StubProvider(), registry)
        plan = runner.plan(task)
    except ValidationError as exc:
        print(f"agentkit: invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print(f"valid: task {task.text!r}")
    print(f"  tools: {', '.join(t['name'] for t in plan['tools'])}")
    print(f"  max steps: {plan['max_steps']}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentkit",
        description="Dependency-free LLM agent runner (stdlib only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a task file.")
    run.add_argument("task_file", help="JSON task file (see examples/).")
    run.add_argument(
        "--provider",
        default="stub",
        choices=["stub", "openai"],
        help="LLM provider (default: stub, deterministic, offline).",
    )
    run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the task file's step limit.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing any tool.",
    )
    run.add_argument("--trace", default=None, help="Write the JSON trace here.")
    run.add_argument("--report", default=None, help="Write the text report here.")
    run.add_argument(
        "--notes-dir", default=None, help="Directory of markdown notes for search_notes."
    )
    run.add_argument(
        "--base-dir",
        default=None,
        help="Directory read_file is confined to (default: working directory).",
    )
    run.add_argument(
        "--allow-shell",
        action="store_true",
        help="Register the opt-in shell tool (logs a warning per use).",
    )
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="Validate a task file and tools.")
    validate.add_argument("task_file", help="JSON task file (see examples/).")
    validate.add_argument("--notes-dir", default=None)
    validate.add_argument("--base-dir", default=None)
    validate.add_argument("--allow-shell", action="store_true")
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
