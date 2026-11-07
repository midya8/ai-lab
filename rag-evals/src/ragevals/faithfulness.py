"""Answer-faithfulness evaluation: is every claim in an answer supported by retrieved context?

Dependency-free lexical approach (a stand-in for an NLI/LLM judge, which can be
plugged in through the `supports` callable):
  1. split the answer into claims (sentences),
  2. a claim is supported if enough of its content words (and numbers) appear
     in a single context chunk,
  3. faithfulness = supported claims / total claims.
"""
from __future__ import annotations
import re
from typing import Callable, Sequence

_STOP = set("a an the of to in on for and or is are was were be been it its this that these those with as by at from "
            "which what how why does do did can could should would will not no than then so such into over also".split())
_SENT = re.compile(r"(?<=[.!?])\s+")
_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def content_words(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in _SENT.split(answer.strip()) if len(content_words(s)) >= 2]


def claim_support(claim: str, context: Sequence[str]) -> float:
    """Best fraction of the claim's content words covered by one context passage."""
    words = content_words(claim)
    if not words:
        return 1.0
    return max((len(words & content_words(c)) / len(words) for c in context), default=0.0)


def faithfulness(answer: str, context: Sequence[str], threshold: float = 0.6,
                 supports: Callable[[str, Sequence[str]], bool] | None = None) -> dict:
    claims = split_claims(answer)
    if not claims:
        return {"score": 1.0, "claims": [], "unsupported": []}
    rows = []
    for c in claims:
        ok = supports(c, context) if supports else claim_support(c, context) >= threshold
        rows.append({"claim": c, "support": round(claim_support(c, context), 3), "supported": bool(ok)})
    bad = [r["claim"] for r in rows if not r["supported"]]
    return {"score": 1 - len(bad) / len(rows), "claims": rows, "unsupported": bad}


def answer_relevance(question: str, answer: str) -> float:
    """Fraction of the question's content words addressed by the answer."""
    q = content_words(question)
    return len(q & content_words(answer)) / len(q) if q else 0.0


def evaluate_answers(items: list[dict], threshold: float = 0.6) -> dict:
    """items: {"question", "answer", "context": [str]}"""
    if not items:
        raise ValueError("empty items")
    res = [faithfulness(i["answer"], i["context"], threshold) for i in items]
    rel = [answer_relevance(i["question"], i["answer"]) for i in items]
    n = len(items)
    return {"n": n, "faithfulness": sum(r["score"] for r in res) / n,
            "answer_relevance": sum(rel) / n,
            "hallucinated_answers": sum(1 for r in res if r["unsupported"])}
