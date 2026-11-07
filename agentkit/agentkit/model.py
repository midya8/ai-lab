"""Data model and validation for agentkit.

Everything in this module is plain data plus validation. No I/O, no
network, no LLM calls: a bad task or a bad tool definition is rejected
here, before anything runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# JSON Schema primitive types we validate arguments against.
_SCHEMA_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Terminal statuses for a run. "success" and "dry_run" exit 0;
# everything else exits 1 (agent failure) or 2 (invalid input).
STATUSES = (
    "success",
    "failed",
    "step_limit",
    "loop_detected",
    "timeout",
    "dry_run",
)


class AgentError(Exception):
    """Base error for agentkit."""


class ValidationError(AgentError):
    """Raised for invalid tasks or tool definitions (exit code 2)."""


class ToolExecutionError(AgentError):
    """Raised when a tool fails while running (exit code 1)."""


@dataclass
class Tool:
    """A tool the agent may call.

    ``parameters`` is a JSON Schema object describing the arguments:
    ``{"type": "object", "properties": {...}, "required": [...]}``.
    ``func`` is the Python callable invoked with the validated
    arguments as keyword arguments.
    """

    name: str
    description: str
    parameters: dict
    func: object = field(repr=False)

    def validate(self) -> None:
        """Validate the tool definition itself. Raises ValidationError."""
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ValidationError(
                f"invalid tool name {self.name!r}: must start with a letter "
                "and contain only letters, digits and underscores"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValidationError(f"tool {self.name!r} needs a non-empty description")
        if not isinstance(self.parameters, dict):
            raise ValidationError(f"tool {self.name!r}: parameters must be an object")
        if self.parameters.get("type") != "object":
            raise ValidationError(
                f"tool {self.name!r}: parameters.type must be 'object'"
            )
        properties = self.parameters.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValidationError(
                f"tool {self.name!r}: parameters.properties must be a "
                "non-empty object"
            )
        for prop, schema in properties.items():
            if not isinstance(schema, dict) or schema.get("type") not in _SCHEMA_TYPES:
                raise ValidationError(
                    f"tool {self.name!r}: property {prop!r} has no valid "
                    f"type (got {schema!r})"
                )
        required = self.parameters.get("required", [])
        if not isinstance(required, list) or any(
            r not in properties for r in required
        ):
            raise ValidationError(
                f"tool {self.name!r}: required lists an unknown property"
            )
        if not callable(self.func):
            raise ValidationError(f"tool {self.name!r}: func is not callable")

    def validate_args(self, args: object) -> dict:
        """Validate a call's arguments against the schema.

        Returns the arguments as a dict. Raises ValidationError on any
        mismatch: non-object args, missing required fields, unknown
        fields, or wrong types. ``bool`` is not accepted as ``integer``
        (in Python ``bool`` is a subclass of ``int``).
        """
        if not isinstance(args, dict):
            raise ValidationError(
                f"tool {self.name!r}: arguments must be an object, "
                f"got {type(args).__name__}"
            )
        properties = self.parameters["properties"]
        for key in args:
            if key not in properties:
                raise ValidationError(
                    f"tool {self.name!r}: unknown argument {key!r}"
                )
        for key in self.parameters.get("required", []):
            if key not in args:
                raise ValidationError(
                    f"tool {self.name!r}: missing required argument {key!r}"
                )
        for key, value in args.items():
            expected = _SCHEMA_TYPES[properties[key]["type"]]
            if properties[key]["type"] == "integer" and isinstance(value, bool):
                raise ValidationError(
                    f"tool {self.name!r}: argument {key!r} must be an "
                    f"integer, got boolean"
                )
            if not isinstance(value, expected):
                raise ValidationError(
                    f"tool {self.name!r}: argument {key!r} must be "
                    f"{properties[key]['type']}, got {type(value).__name__}"
                )
        return args

    def schema(self) -> dict:
        """The JSON Schema sent to the provider for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    """One tool invocation requested by the provider."""

    name: str
    arguments: dict

    def canonical(self) -> str:
        """A stable string identifying this exact call (for loop detection)."""
        return json.dumps(
            {"name": self.name, "arguments": self.arguments},
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass
class StepRecord:
    """One step of a run, as written to the trace file."""

    step: int
    provider_text: str | None = None
    tool_call: ToolCall | None = None
    tool_result: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "provider_text": self.provider_text,
            "tool_call": (
                {"name": self.tool_call.name, "arguments": self.tool_call.arguments}
                if self.tool_call
                else None
            ),
            "tool_result": self.tool_result,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class RunResult:
    """The outcome of one agent run."""

    status: str
    task: str
    steps: list = field(default_factory=list)
    final_answer: str | None = None
    duration_seconds: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in ("success", "dry_run")


@dataclass
class Task:
    """A task file: what to do, with what tools, under what limits."""

    text: str
    max_steps: int = 10
    tools: list | None = None  # None = all registered tools
    notes_dir: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        if not isinstance(data, dict):
            raise ValidationError("task file must contain a JSON object")
        text = data.get("task")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("task file needs a non-empty string 'task'")
        max_steps = data.get("max_steps", 10)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool):
            raise ValidationError("'max_steps' must be an integer")
        if max_steps < 1 or max_steps > 100:
            raise ValidationError("'max_steps' must be between 1 and 100")
        tools = data.get("tools")
        if tools is not None and (
            not isinstance(tools, list)
            or any(not isinstance(t, str) for t in tools)
        ):
            raise ValidationError("'tools' must be a list of tool names")
        timeout = data.get("timeout_seconds", 30.0)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValidationError("'timeout_seconds' must be a number")
        if timeout <= 0 or timeout > 600:
            raise ValidationError("'timeout_seconds' must be between 0 and 600")
        notes_dir = data.get("notes_dir")
        if notes_dir is not None and not isinstance(notes_dir, str):
            raise ValidationError("'notes_dir' must be a string path")
        return cls(
            text=text.strip(),
            max_steps=max_steps,
            tools=tools,
            notes_dir=notes_dir,
            timeout_seconds=float(timeout),
        )

    @classmethod
    def from_file(cls, path: str) -> "Task":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise ValidationError(f"task file not found: {path}")
        except json.JSONDecodeError as exc:
            raise ValidationError(f"task file {path} is not valid JSON: {exc}")
        except OSError as exc:
            raise ValidationError(f"cannot read task file {path}: {exc}")
        return cls.from_dict(data)
