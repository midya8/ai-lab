"""Tool registry and built-in tools for agentkit.

Tools are plain Python functions declared with a JSON Schema. The
registry validates every definition up front and rejects unknown tool
calls at run time — a provider can never execute a tool that was not
registered. No tool runs a shell command unless the caller explicitly
opts in to the shell tool.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import warnings

from .model import Tool, ToolExecutionError, ValidationError

MAX_FILE_BYTES = 1_000_000
MAX_SEARCH_RESULTS = 20


# ---------------------------------------------------------------------------
# Calculator: safe arithmetic via the AST, never eval().
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(left) > 1000 or abs(right) > 100:
                raise ToolExecutionError("calculator: exponent too large")
            return left ** right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ToolExecutionError(
        "calculator: only numbers and + - * / // % ** ( ) are allowed"
    )


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolExecutionError("calculator: expression must be a non-empty string")
    if len(expression) > 200:
        raise ToolExecutionError("calculator: expression too long (max 200 chars)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(f"calculator: invalid expression: {exc}")
    try:
        result = _eval_node(tree)
    except ZeroDivisionError:
        raise ToolExecutionError("calculator: division by zero")
    except ToolExecutionError:
        raise
    except (OverflowError, ValueError) as exc:
        raise ToolExecutionError(f"calculator: {exc}")
    return f"{expression.strip()} = {result}"


# ---------------------------------------------------------------------------
# File reader: sandboxed to a base directory.
# ---------------------------------------------------------------------------

def make_read_file(base_dir: str):
    """Build a read_file tool confined to ``base_dir``.

    Absolute paths and ``..`` escapes are rejected; symlinks that
    point outside the base are rejected too.
    """
    base = os.path.realpath(base_dir)

    def read_file(path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ToolExecutionError("read_file: path must be a non-empty string")
        target = os.path.realpath(os.path.join(base, path))
        if os.path.commonpath([base, target]) != base:
            raise ToolExecutionError(
                f"read_file: {path!r} is outside the allowed directory"
            )
        if not os.path.isfile(target):
            raise ToolExecutionError(f"read_file: no such file: {path!r}")
        if os.path.getsize(target) > MAX_FILE_BYTES:
            raise ToolExecutionError(
                f"read_file: {path!r} exceeds the 1 MB size limit"
            )
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError as exc:
            raise ToolExecutionError(f"read_file: cannot read {path!r}: {exc}")

    return read_file


# ---------------------------------------------------------------------------
# search_notes: substring search over local markdown files.
# ---------------------------------------------------------------------------

def make_search_notes(notes_dir: str):
    """Build a search_notes tool over ``*.md`` files under ``notes_dir``."""
    base = os.path.realpath(notes_dir)

    def search_notes(query: str, max_results: int = 5) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ToolExecutionError("search_notes: query must be a non-empty string")
        if not isinstance(max_results, int) or max_results < 1:
            raise ToolExecutionError("search_notes: max_results must be >= 1")
        max_results = min(max_results, MAX_SEARCH_RESULTS)
        if not os.path.isdir(base):
            raise ToolExecutionError(
                f"search_notes: notes directory not found: {notes_dir!r}"
            )
        needle = query.strip().lower()
        hits: list[str] = []
        for root, _dirs, files in os.walk(base):
            for name in sorted(files):
                if not name.endswith(".md"):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, base)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        lines = fh.readlines()
                except OSError:
                    continue
                for lineno, line in enumerate(lines, start=1):
                    if needle in line.lower():
                        hits.append(f"{rel}:{lineno}: {line.strip()}")
                        if len(hits) >= max_results:
                            return "\n".join(hits)
        if not hits:
            return f"No matches for {query!r} in {notes_dir!r}."
        return "\n".join(hits)

    return search_notes


# ---------------------------------------------------------------------------
# Shell tool: opt-in only, always logs a warning when it runs.
# ---------------------------------------------------------------------------

def shell(command: str) -> str:
    """Run a shell command and return its combined output."""
    warnings.warn(
        "agentkit: shell tool executed a command — this is opt-in only",
        UserWarning,
        stacklevel=2,
    )
    if not isinstance(command, str) or not command.strip():
        raise ToolExecutionError("shell: command must be a non-empty string")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ToolExecutionError("shell: command timed out after 30s")
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise ToolExecutionError(
            f"shell: command exited {proc.returncode}: {output[:500]}"
        )
    return output if output else "(no output)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Holds the tools an agent run may use."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        tool.validate()
        if tool.name in self._tools:
            raise ValidationError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def validate_all(self) -> None:
        for tool in self._tools.values():
            tool.validate()

    def schemas(self) -> list[dict]:
        return [self._tools[name].schema() for name in self.names()]

    def call(self, name: str, arguments: dict) -> str:
        """Validate and execute a tool call. Unknown tools are rejected."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValidationError(f"unknown tool: {name!r}")
        tool.validate_args(arguments)
        result = tool.func(**arguments)
        return result if isinstance(result, str) else str(result)


def builtin_tools(
    notes_dir: str | None = None,
    base_dir: str | None = None,
    allow_shell: bool = False,
) -> ToolRegistry:
    """The built-in tool set: calculator, read_file, search_notes.

    ``base_dir`` confines read_file (defaults to the working directory);
    ``notes_dir`` is where search_notes looks (defaults to ``base_dir``).
    The shell tool is only registered when ``allow_shell`` is True.
    """
    base = base_dir or os.getcwd()
    notes = notes_dir or base
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description=(
                "Evaluate an arithmetic expression with + - * / // % ** and "
                "parentheses, e.g. '17 + 25' or '(3 + 4) * 2'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read a text file inside the allowed directory and return "
                "its contents. Paths outside the directory are rejected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the allowed directory.",
                    }
                },
                "required": ["path"],
            },
            func=make_read_file(base),
        )
    )
    registry.register(
        Tool(
            name="search_notes",
            description=(
                "Search local markdown notes for a phrase and return "
                "matching lines with file and line number."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Phrase to search for (case-insensitive).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matches to return (default 5).",
                    },
                },
                "required": ["query"],
            },
            func=make_search_notes(notes),
        )
    )
    if allow_shell:
        registry.register(
            Tool(
                name="shell",
                description=(
                    "Run a shell command. Opt-in only: this tool is never "
                    "registered by default, and every execution is logged "
                    "with a warning."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run.",
                        }
                    },
                    "required": ["command"],
                },
                func=shell,
            )
        )
    return registry
