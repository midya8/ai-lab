import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "agentkit"))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

import pytest  # noqa: E402

from agentkit.model import ToolExecutionError, ValidationError  # noqa: E402
from agentkit.tools import builtin_tools  # noqa: E402
from docqa.agent_tool import register  # noqa: E402


@pytest.fixture(scope="module")
def reg():
    return register(builtin_tools(base_dir=str(DOCS)), DOCS)


def test_registered_next_to_builtins(reg):
    assert "ask_docs" in reg.names() and "calculator" in reg.names()


def test_cited_answer(reg):
    out = reg.call("ask_docs", {"question": "How long do I have to return a bike?"})
    assert "30 days" in out and "[returns.md:" in out


def test_abstains_off_topic(reg):
    out = reg.call("ask_docs", {"question": "What is the capital of Mongolia?"})
    assert out.startswith("I don't know")


def test_rejects_bad_args(reg):
    with pytest.raises((ValidationError, ToolExecutionError)):
        reg.call("ask_docs", {"question": 5})
    with pytest.raises(ToolExecutionError):
        reg.call("ask_docs", {"question": "   "})


def test_missing_dir():
    with pytest.raises(ToolExecutionError):
        register(builtin_tools(base_dir=str(DOCS)), DOCS / "nope")
