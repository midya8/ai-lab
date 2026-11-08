import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

import pytest  # noqa: E402

from docqa import DocQA  # noqa: E402
from docqa.__main__ import main  # noqa: E402
from docqa.evaluate import run_eval  # noqa: E402


@pytest.fixture(scope="module")
def qa():
    return DocQA.from_dir(DOCS)


def test_answers_with_citation(qa):
    r = qa.ask("How long do I have to return a bike?")
    assert r.answered and "30 days" in r.text
    assert r.evidence[0].citation.startswith("[returns.md:")


def test_sentences_are_verbatim(qa):
    text = " ".join(p.read_text() for p in DOCS.glob("*.md"))
    flat = " ".join(text.split())
    r = qa.ask("Is there a restocking fee for unused items?")
    assert all(e.sentence in flat for e in r.evidence)


@pytest.mark.parametrize("q", ["What is the capital of France?", "Who is the CEO of Harbour & Spoke?"])
def test_abstains(qa, q):
    r = qa.ask(q)
    assert not r.answered and r.text.startswith("I don't know")


def test_empty_question(qa):
    assert not qa.ask("the of and").answered


def test_bad_coverage():
    with pytest.raises(ValueError):
        DocQA([], min_coverage=2)


def test_strict_threshold_abstains():
    assert not DocQA.from_dir(DOCS, min_coverage=1.0).ask("How fast do refunds arrive?").answered


def test_eval(qa):
    res = run_eval(qa, ROOT / "eval" / "questions.json")
    assert res["answer_rate"] == 1.0 and res["abstain_rate"] == 1.0


def test_cli(capsys):
    assert main(["ask", str(DOCS), "How long do I have to return a bike?", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answered"]
    assert main(["ask", str(DOCS), "What is the capital of France?"]) == 1
    assert main(["ask", "/nonexistent", "x"]) == 2


def test_paraphrase_eval(qa):
    res = run_eval(qa, ROOT / "eval" / "paraphrase.json")
    assert res["answer_rate"] == 1.0 and res["abstain_rate"] == 1.0, res["failures"]


def test_paraphrase_off_is_worse():
    strict = DocQA.from_dir(DOCS, paraphrase=False)
    res = run_eval(strict, ROOT / "eval" / "paraphrase.json")
    assert res["answer_rate"] < 1.0


def test_paraphrase_does_not_break_original_eval():
    res = run_eval(DocQA.from_dir(DOCS), ROOT / "eval" / "questions.json")
    assert res["answer_rate"] == 1.0 and res["abstain_rate"] == 1.0


def test_broad_eval_yes_no_fixed(qa):
    res = run_eval(qa, ROOT / "eval" / "broad.json")
    assert res["abstain_rate"] == 1.0
    assert res["answer_rate"] == 1.0, res["failures"]


def test_yes_no_questions_cite_evidence(qa):
    r = qa.ask("Is the shop open on Sundays?")
    assert r.answered and "closed on Sundays" in r.text
    r = qa.ask("Is shipping free over a certain amount?")
    assert r.answered and "$150" in r.text


def test_yes_no_still_abstains_when_unsupported(qa):
    assert not qa.ask("Is the shop open on Mars?").answered
    assert not qa.ask("Is there a certain amount of snow?").answered
