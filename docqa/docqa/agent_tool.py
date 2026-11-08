"""Expose DocQA as an agentkit tool.

The tool returns a cited answer or an explicit abstention, so an agent
can ground itself in documents instead of guessing. agentkit is imported
lazily (installed package, else the sibling folder in the monorepo).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .qa import DocQA


def _agentkit():
    try:
        import agentkit  # noqa: F401
    except ImportError:  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agentkit"))
    from agentkit.model import Tool, ToolExecutionError
    return Tool, ToolExecutionError


def make_docqa_tool(docs_dir: str | Path, **qa_kwargs):
    """Build an agentkit ``Tool`` named ``ask_docs`` over a docs folder."""
    Tool, ToolExecutionError = _agentkit()
    if not Path(docs_dir).is_dir():
        raise ToolExecutionError(f"ask_docs: docs directory not found: {str(docs_dir)!r}")
    qa = DocQA.from_dir(docs_dir, **qa_kwargs)

    def ask_docs(question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ToolExecutionError("ask_docs: question must be a non-empty string")
        return qa.ask(question).text

    return Tool(
        name="ask_docs",
        description=(
            "Answer a question from the policy documents. Returns verbatim "
            "sentences with [file:chunk] citations, or says it doesn't know."
        ),
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The question to answer."}},
            "required": ["question"],
        },
        func=ask_docs,
    )


def register(registry, docs_dir: str | Path, **qa_kwargs):
    """Add ``ask_docs`` to an agentkit ToolRegistry and return it."""
    registry.register(make_docqa_tool(docs_dir, **qa_kwargs))
    return registry
