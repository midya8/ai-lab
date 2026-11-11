"""Generative answers over DocQA evidence, routed and grounding-checked.

DocQA's default answer is extractive: it quotes the best-supported sentences.
``GroundedGenerator`` adds an optional *generative* stage on top: the retrieved
evidence is put into a prompt, an LLM writes the answer, and a **grounding
verifier** decides whether that answer may be shown:

* every number in the answer must appear in the evidence (hallucinated figures
  are the commonest failure), and
* every sentence's content words must be covered >= ``min_support`` by the
  evidence text.

The generator is duck-typed to llm-router's ``Router``: anything with
``route(prompt, max_tokens=..., verifier=...)`` returning an object with
``text``, ``model``, ``cost`` and ``verified``. With a real Router the verifier
turns into a *cascade*: the cheapest model answers first and the router
escalates to a stronger model only while the answer fails the grounding check.
If no model produces a grounded answer, the generator falls back to DocQA's
extractive answer (which is grounded by construction) instead of showing an
unsupported one. If DocQA finds no evidence it abstains without calling a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragkit.retrieve import tokenize

from .qa import FILLER, SPLIT, DocQA, QAResult

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
ABSTAIN_RE = re.compile(r"\bi (?:don't|do not) know\b", re.I)
CITE_RE = re.compile(r"\[[^\]]*\]")

PROMPT = """Answer the question using ONLY the evidence below. Cite sources in brackets.
If the evidence does not contain the answer, say "I don't know".

Evidence:
{evidence}

Question: {question}
Answer:"""


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUM_RE.findall(text)}


def _content(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in FILLER and len(t) > 1}


def grounding_report(answer: str, evidence_text: str, min_support: float = 0.7) -> dict:
    """Explain how well ``answer`` is supported by ``evidence_text``."""
    body = CITE_RE.sub(" ", answer)
    ev_terms = set(tokenize(evidence_text))
    ev_nums = _numbers(evidence_text)
    unsupported_numbers = sorted(_numbers(body) - ev_nums)
    weak: list[tuple[str, float]] = []
    worst = 1.0
    for sent in SPLIT.split(body):
        terms = _content(sent)
        if not terms:
            continue
        support = len(terms & ev_terms) / len(terms)
        worst = min(worst, support)
        if support < min_support:
            weak.append((sent.strip(), round(support, 3)))
    return {"grounded": not unsupported_numbers and not weak,
            "unsupported_numbers": unsupported_numbers,
            "weak_sentences": weak, "min_support": round(worst, 3)}


@dataclass
class GeneratedAnswer:
    question: str
    text: str
    answered: bool
    source: str  # "generated" | "extractive-fallback" | "abstained"
    model: str | None = None
    cost: float = 0.0
    escalations: int = 0
    evidence: list = field(default_factory=list)


class GroundedGenerator:
    def __init__(self, qa: DocQA, router, min_support: float = 0.7, max_tokens: int = 200):
        if not 0 < min_support <= 1:
            raise ValueError("min_support must be in (0, 1]")
        self.qa = qa
        self.router = router
        self.min_support = min_support
        self.max_tokens = max_tokens

    def _evidence_block(self, res: QAResult) -> str:
        return "\n".join(f"- {e.sentence} {e.citation}" for e in res.evidence)

    def ask(self, question: str) -> GeneratedAnswer:
        res = self.qa.ask(question)
        if not res.answered:
            return GeneratedAnswer(question, res.text, False, "abstained")
        block = self._evidence_block(res)
        prompt = PROMPT.format(evidence=block, question=question)
        ev_text = " ".join(e.sentence for e in res.evidence)

        def verifier(text: str) -> bool:
            if ABSTAIN_RE.search(text):
                return False  # evidence exists, so an abstention is not acceptable here
            return grounding_report(text, ev_text, self.min_support)["grounded"]

        try:
            out = self.router.route(prompt, max_tokens=self.max_tokens, verifier=verifier)
        except Exception:  # all models failed / over budget: stay useful and safe
            return GeneratedAnswer(question, res.text, True, "extractive-fallback",
                                   evidence=res.evidence)
        escalations = getattr(out, "fallbacks", 0)
        if getattr(out, "verified", None) is True:
            return GeneratedAnswer(question, out.text.strip(), True, "generated",
                                   out.model, out.cost, escalations, res.evidence)
        return GeneratedAnswer(question, res.text, True, "extractive-fallback",
                               out.model, out.cost, escalations, res.evidence)
