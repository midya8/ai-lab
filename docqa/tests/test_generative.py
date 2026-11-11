import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "llm-router"))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"
MODELS = ROOT.parent / "llm-router" / "examples" / "models.json"

import pytest  # noqa: E402

from llmrouter.models import load_models  # noqa: E402
from llmrouter.providers import Completion, ProviderError  # noqa: E402
from llmrouter.router import Router  # noqa: E402

from docqa.generative import GroundedGenerator, grounding_report  # noqa: E402
from docqa.qa import DocQA  # noqa: E402

EV = "Refunds are issued within 14 days of the request. Contact billing for help."


def test_report_flags_invented_number():
    r = grounding_report("Refunds are issued within 30 days.", EV)
    assert not r["grounded"] and r["unsupported_numbers"] == ["30"]


def test_report_accepts_supported_answer_and_ignores_citations():
    r = grounding_report("Refunds are issued within 14 days of the request. [policy.md]", EV)
    assert r["grounded"] and r["min_support"] == 1.0


def test_report_flags_unrelated_sentence():
    r = grounding_report("The moon orbits the earth every month.", EV)
    assert not r["grounded"] and r["weak_sentences"]


class ByModel:
    """Provider whose answer depends on the model name; records calls."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def complete(self, model, prompt, max_tokens):
        self.calls.append(model.name)
        a = self.answers[model.name]
        if isinstance(a, Exception):
            raise a
        return Completion(a, 200, 30, 100.0)


class Stub:  # a router stand-in that is not llm-router
    def __init__(self, text, verified=True):
        self.text, self.verified, self.calls = text, verified, 0

    def route(self, prompt, max_tokens=0, verifier=None):
        self.calls += 1
        ok = verifier(self.text)

        class R:
            pass
        r = R()
        r.text, r.model, r.cost, r.fallbacks, r.verified = self.text, "stub", 0.001, 0, ok
        return r


@pytest.fixture(scope="module")
def qa():
    return DocQA.from_dir(DOCS)


def _qa_and_question(qa):
    res = None
    for q in ["How long does a refund take?", "How do I reset my password?",
              "What is the retention period for logs?"]:
        r = qa.ask(q)
        if r.answered:
            return q, r
    pytest.skip("no answerable sample question in ragkit docs")


def test_abstains_without_calling_model(qa):
    stub = Stub("anything")
    g = GroundedGenerator(qa, stub)
    out = g.ask("What is the airspeed velocity of an unladen swallow?")
    assert out.source == "abstained" and not out.answered and stub.calls == 0


def test_grounded_generation_is_returned(qa):
    q, res = _qa_and_question(qa)
    good = " ".join(e.sentence for e in res.evidence)
    out = GroundedGenerator(qa, Stub(good)).ask(q)
    assert out.source == "generated" and out.text == good.strip()


def test_ungrounded_generation_falls_back_to_extractive(qa):
    q, res = _qa_and_question(qa)
    out = GroundedGenerator(qa, Stub("It takes 9999 days, per cosmic law.")).ask(q)
    assert out.source == "extractive-fallback" and out.text == res.text


def test_abstaining_model_is_rejected_when_evidence_exists(qa):
    q, res = _qa_and_question(qa)
    out = GroundedGenerator(qa, Stub("I don't know")).ask(q)
    assert out.source == "extractive-fallback"


def test_cascade_escalates_past_hallucinating_cheap_model(qa):
    q, res = _qa_and_question(qa)
    models = sorted(load_models(MODELS), key=lambda m: m.quality)
    good = " ".join(e.sentence for e in res.evidence)
    answers = {m.name: "Invented figure 424242 here." for m in models}
    answers[models[-1].name] = good
    prov = ByModel(answers)
    out = GroundedGenerator(qa, Router(models, prov)).ask(q)
    assert out.source == "generated" and out.model == models[-1].name
    assert out.escalations >= 1 and len(prov.calls) >= 2 and out.cost > 0


def test_all_models_failing_falls_back(qa):
    q, res = _qa_and_question(qa)
    models = load_models(MODELS)
    prov = ByModel({m.name: ProviderError("down") for m in models})
    out = GroundedGenerator(qa, Router(models, prov)).ask(q)
    assert out.source == "extractive-fallback" and out.text == res.text


def test_bad_min_support(qa):
    with pytest.raises(ValueError):
        GroundedGenerator(qa, Stub("x"), min_support=0)
