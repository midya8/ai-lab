"""Sentence-level grounded QA built on ragkit's retriever.

Every sentence in an answer is copied verbatim from a document and
carries a citation. If the best sentence does not cover enough of the
question's content terms, docqa abstains instead of guessing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.ingest import load_documents
from ragkit.retrieve import Retriever, tokenize

SPLIT = re.compile(r"(?<=[.!?])\s+")
# question words that carry the answer's *type*, not content to look up
FILLER = {"long", "fast", "quickly", "much", "many", "often", "soon", "late", "early"}
QUANTITY_RE = re.compile(r"\b(how\s+(long|many|much|fast|often|soon|late|early)|when|what\s+time)\b", re.I)
DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class Evidence:
    sentence: str
    citation: str
    coverage: float


@dataclass
class QAResult:
    question: str
    answered: bool
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def text(self) -> str:
        if not self.answered:
            return "I don't know: the documents don't contain enough to answer that."
        return " ".join(f"{e.sentence} {e.citation}" for e in self.evidence)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answered": self.answered,
            "confidence": round(self.confidence, 3),
            "answer": self.text,
            "evidence": [e.__dict__ for e in self.evidence],
        }


class DocQA:
    def __init__(self, chunks, min_coverage: float = 0.6, max_sentences: int = 2,
                 top_k: int = 4):
        if not 0 <= min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        self.min_coverage = min_coverage
        self.max_sentences = max_sentences
        self.top_k = top_k
        self.retriever = Retriever(list(chunks))
        n = max(len(self.retriever.chunks), 1)
        self._idf = {t: math.log(1 + n / df) for t, df in self.retriever._df.items()}

    @classmethod
    def from_dir(cls, docs_dir: str | Path, **kw) -> "DocQA":
        return cls(load_documents(docs_dir), **kw)

    def _weight(self, term: str) -> float:
        # unseen terms get the highest idf: they are the likeliest to be unanswerable
        return self._idf.get(term, math.log(1 + len(self.retriever.chunks)))

    def coverage(self, q_terms: set[str], sentence: str, quantity: bool = False) -> float:
        score = self._lexical(q_terms, sentence)
        if quantity:  # the question wants a number or time: favour sentences that have one
            score = min(1.0, score + 0.3) if DIGIT_RE.search(sentence) else score * 0.7
        return score

    def _lexical(self, q_terms: set[str], sentence: str) -> float:
        total = sum(self._weight(t) for t in q_terms)
        if total == 0:
            return 0.0
        hit = sum(self._weight(t) for t in q_terms & set(tokenize(sentence)))
        return hit / total

    def ask(self, question: str) -> QAResult:
        quantity = bool(QUANTITY_RE.search(question))
        q_terms = {t for t in tokenize(question) if t not in FILLER}
        if not q_terms:
            return QAResult(question, False)
        cands: list[Evidence] = []
        seen: set[str] = set()
        for sc in self.retriever.search(question, k=self.top_k):
            for s in SPLIT.split(sc.chunk.text):
                s = " ".join(s.split())
                if len(s) < 8 or s in seen:
                    continue
                seen.add(s)
                cands.append(Evidence(s, sc.chunk.citation(), self.coverage(q_terms, s, quantity)))
        cands.sort(key=lambda e: -e.coverage)
        if not cands or cands[0].coverage < self.min_coverage:
            return QAResult(question, False, confidence=cands[0].coverage if cands else 0.0)
        best = cands[0]
        picked = [best]
        covered = q_terms & set(tokenize(best.sentence))
        for e in cands[1:]:
            if len(picked) >= self.max_sentences:
                break
            new = (q_terms & set(tokenize(e.sentence))) - covered
            if new and e.coverage >= self.min_coverage / 2:
                picked.append(e)
                covered |= new
        return QAResult(question, True, picked, best.coverage)
