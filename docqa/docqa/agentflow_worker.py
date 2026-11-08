"""Use DocQA as a worker inside an agentflow DAG.

``DocQAWorker`` is duck-typed to agentflow's ``Agent`` (``name`` + ``run(prompt)``),
so it drops straight into an ``Orchestrator``. With ``strict=True`` an abstention
raises ``Abstained``: the task fails (after its retries) and downstream tasks are
marked skipped, so a pipeline never builds on an unsupported answer.
"""
from __future__ import annotations

from pathlib import Path

from .qa import DocQA


class Abstained(RuntimeError):
    """The documents did not support an answer."""


class DocQAWorker:
    def __init__(self, docs_dir: str | Path, name: str = "docqa", strict: bool = False, **qa_kwargs):
        if not Path(docs_dir).is_dir():
            raise FileNotFoundError(f"docs directory not found: {str(docs_dir)!r}")
        self.name = name
        self.strict = strict
        self.qa = DocQA.from_dir(docs_dir, **qa_kwargs)
        self.calls = 0

    def run(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("DocQAWorker: prompt must be a non-empty question")
        self.calls += 1
        res = self.qa.ask(prompt.strip())
        if not res.answered and self.strict:
            raise Abstained(f"no supported answer for: {prompt.strip()!r}")
        return res.text
