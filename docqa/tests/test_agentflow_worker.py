import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "agentflow" / "src"))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

import pytest  # noqa: E402

from agentflow import Agent, Orchestrator, ScriptedLLM, Task  # noqa: E402
from docqa.agentflow_worker import Abstained, DocQAWorker  # noqa: E402


def _summariser():
    return Agent("writer", ScriptedLLM([{"final": "summary ok"}]))


def test_worker_in_dag_feeds_downstream():
    seen = []

    class Echo:
        name = "echo"

        def run(self, prompt):
            seen.append(prompt)
            return "done"

    orch = Orchestrator([DocQAWorker(DOCS), Echo()])
    res = orch.run([
        Task("q", "docqa", "How long do I have to return a bike?"),
        Task("s", "echo", "Summarise: {q}", deps=["q"]),
    ])
    assert res["q"].status == "ok" and "30 days" in res["q"].output
    assert res["s"].status == "ok" and "30 days" in seen[0] and "[returns.md:" in seen[0]


def test_lenient_abstention_is_plain_answer():
    out = DocQAWorker(DOCS).run("What is the capital of Mongolia?")
    assert out.startswith("I don't know")


def test_strict_abstention_fails_and_skips_downstream():
    orch = Orchestrator([DocQAWorker(DOCS, strict=True), _summariser()])
    res = orch.run([
        Task("q", "docqa", "What is the capital of Mongolia?", retries=1),
        Task("s", "writer", "Summarise: {q}", deps=["q"]),
    ])
    assert res["q"].status == "failed" and res["q"].attempts == 2
    assert "no supported answer" in res["q"].error
    assert res["s"].status == "skipped"


def test_strict_raises_abstained_directly():
    with pytest.raises(Abstained):
        DocQAWorker(DOCS, strict=True).run("What is the capital of Mongolia?")


def test_bad_inputs():
    with pytest.raises(FileNotFoundError):
        DocQAWorker(DOCS / "nope")
    with pytest.raises(ValueError):
        DocQAWorker(DOCS).run("  ")


def test_parallel_questions_share_one_index():
    w = DocQAWorker(DOCS)
    orch = Orchestrator([w])
    res = orch.run([Task(f"q{i}", "docqa", "How long do I have to return a bike?") for i in range(4)])
    assert all(r.status == "ok" for r in res.values()) and w.calls == 4
