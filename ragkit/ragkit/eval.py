"""Evaluation against a golden set: recall@k and citation precision.

A golden entry is {"question": ..., "expected_sources": [...],
"answer_span": "..."} -- the span is a distinctive phrase the
correct answer must contain. Metrics:

* recall@k: fraction of expected source files present in the top-k
  retrieved chunks.
* citation precision: fraction of the extractive answer's citations
  whose chunk text actually contains the answer span. A citation
  that does not support the answer counts against precision.

A question passes only when recall@k is 1.0 and precision is 1.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .answer import ExtractiveAnswerer
from .chunk import Chunk
from .retrieve import Retriever, select_for_answer


@dataclass
class GoldenQuestion:
    question: str
    expected_sources: list[str]
    answer_span: str = ""


@dataclass
class QuestionResult:
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    recall: float
    precision: float | None
    passed: bool


@dataclass
class EvalSummary:
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def mean_precision(self) -> float:
        precisions = [r.precision for r in self.results if r.precision is not None]
        if not precisions:
            return 0.0
        return sum(precisions) / len(precisions)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_questions(self) -> list[str]:
        return [r.question for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "n_questions": len(self.results),
            "mean_recall_at_k": round(self.mean_recall, 4),
            "mean_citation_precision": round(self.mean_precision, 4),
            "passed": self.passed,
            "failed_questions": self.failed_questions,
            "results": [
                {
                    "question": r.question,
                    "expected_sources": r.expected_sources,
                    "retrieved_sources": r.retrieved_sources,
                    "recall": round(r.recall, 4),
                    "citation_precision": (
                        None if r.precision is None else round(r.precision, 4)
                    ),
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


def load_golden(path: str | Path) -> list[GoldenQuestion]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read golden set {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"golden set must be a JSON list: {path}")
    questions = []
    for item in payload:
        questions.append(
            GoldenQuestion(
                question=item["question"],
                expected_sources=list(item.get("expected_sources", [])),
                answer_span=item.get("answer_span", ""),
            )
        )
    return questions


def run_eval(
    retriever: Retriever,
    golden: list[GoldenQuestion],
    k: int = 5,
    answerer: ExtractiveAnswerer | None = None,
) -> EvalSummary:
    """Run the golden set; returns per-question results and means."""
    answerer = answerer or ExtractiveAnswerer()
    summary = EvalSummary()
    for entry in golden:
        scored = retriever.search(entry.question, k=k)
        retrieved_sources = sorted({s.chunk.source for s in scored})
        expected = set(entry.expected_sources)
        recall = (
            len(expected & set(retrieved_sources)) / len(expected) if expected else 1.0
        )
        chunks = [s.chunk for s in select_for_answer(scored, entry.question)]
        answer = answerer.answer(entry.question, chunks)
        precision: float | None
        if not entry.answer_span:
            precision = None
        elif not answer.citations:
            precision = 0.0
        else:
            chunk_by_key = {
                (c.source, c.start_line, c.end_line): c for c in chunks
            }
            hits = 0
            for citation in answer.citations:
                chunk = chunk_by_key.get(
                    (citation.source, citation.start_line, citation.end_line)
                )
                if chunk and entry.answer_span.lower() in chunk.text.lower():
                    hits += 1
            precision = hits / len(answer.citations)
        summary.results.append(
            QuestionResult(
                question=entry.question,
                expected_sources=sorted(expected),
                retrieved_sources=retrieved_sources,
                recall=recall,
                precision=precision,
                passed=recall == 1.0 and precision == 1.0,
            )
        )
    return summary


def report_text(summary: EvalSummary, k: int = 5) -> str:
    lines = [
        f"eval: {len(summary.results)} questions, k={k}",
        f"mean recall@{k}: {summary.mean_recall:.3f}",
        f"mean citation precision: {summary.mean_precision:.3f}",
        f"passed: {summary.passed}/{len(summary.results)}",
    ]
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        precision = (
            "n/a" if result.precision is None else f"{result.precision:.2f}"
        )
        lines.append(
            f"  [{status}] recall={result.recall:.2f} precision={precision} "
            f"{result.question}"
        )
        if not result.passed:
            lines.append(f"         expected={result.expected_sources} "
                         f"got={result.retrieved_sources}")
    return "\n".join(lines)
