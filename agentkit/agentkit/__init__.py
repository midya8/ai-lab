"""agentkit — a dependency-free Python LLM agent runner.

An agent loop that runs a task in steps (plan -> call tools -> observe
results -> decide the next step), with a bounded step limit, loop
detection, and a JSON trace of every run. Standard library only.

Built November 2026 as a portfolio project. Not client work, and not
work done for any employer.
"""

from .model import (
    AgentError,
    RunResult,
    StepRecord,
    Task,
    Tool,
    ToolCall,
    ValidationError,
)
from .providers import OpenAICompatibleProvider, ProviderResponse, StubProvider
from .runner import AgentRunner
from .tools import ToolRegistry, builtin_tools

__all__ = [
    "AgentError",
    "AgentRunner",
    "OpenAICompatibleProvider",
    "ProviderResponse",
    "RunResult",
    "StepRecord",
    "StubProvider",
    "Task",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ValidationError",
    "builtin_tools",
]

__version__ = "0.1.0"
