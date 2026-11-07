"""Answering: a pluggable answerer interface.

The default answerer is extractive and fully offline: it quotes the
most relevant retrieved chunks with citations, so every claim in an
answer can be traced back to a source line range. An optional
OpenAI-compatible answerer implements the same protocol and is only
used when explicitly requested -- it reads its key from the
environment and is never called by the tests.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .chunk import Chunk

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    source: str
    start_line: int
    end_line: int

    def format(self) -> str:
        return f"[{self.source}:{self.start_line}-{self.end_line}]"


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)


class Answerer(Protocol):
    """Anything that turns retrieved chunks into an answer."""

    def answer(self, query: str, chunks: list[Chunk]) -> Answer: ...


def _first_sentences(text: str, max_chars: int = 400) -> str:
    """First sentences of *text*, trimmed at a sentence boundary."""
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    out: list[str] = []
    total = 0
    for sentence in sentences:
        added = len(sentence) + (1 if out else 0)
        if out and total + added > max_chars:
            break
        out.append(sentence)
        total += added
    return " ".join(out) if out else text[:max_chars]


class ExtractiveAnswerer:
    """Deterministic offline answerer: quote, don't invent.

    Returns the most relevant chunks (up to *max_chunks*) as quoted
    excerpts, each with a [source:start-end] citation. No model,
    no network, no API key -- the same input always gives the same
    answer.
    """

    def __init__(self, max_chunks: int = 3, max_chars: int = 400):
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def answer(self, query: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(
                text="I couldn't find anything relevant in the indexed documents.",
                citations=[],
            )
        parts: list[str] = []
        citations: list[Citation] = []
        for chunk in chunks[: self.max_chunks]:
            citation = Citation(chunk.source, chunk.start_line, chunk.end_line)
            citations.append(citation)
            parts.append(f"{citation.format()} {_first_sentences(chunk.text, self.max_chars)}")
        return Answer(
            text="Based on the indexed documents:\n\n" + "\n\n".join(parts),
            citations=citations,
        )


class OpenAICompatibleAnswerer:
    """LLM answerer behind the same Answerer protocol.

    Talks to any OpenAI-compatible ``/chat/completions`` endpoint
    with urllib. The API key comes from RAGKIT_API_KEY (or
    OPENAI_API_KEY) only -- it is never read from a file and never
    logged. Citations are the retrieved chunks the prompt was built
    from, so the answer stays traceable.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = (
            api_key or os.environ.get("RAGKIT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("RAGKIT_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    def answer(self, query: str, chunks: list[Chunk]) -> Answer:
        if not self.api_key:
            raise RuntimeError(
                "No API key found: set RAGKIT_API_KEY (or OPENAI_API_KEY)."
            )
        if not chunks:
            return Answer(
                text="I couldn't find anything relevant in the indexed documents.",
                citations=[],
            )
        context = "\n\n".join(
            f"{c.citation()} {c.text}" for c in chunks[:3]
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "Cite sources with their [file:lines] markers.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        citations = [
            Citation(c.source, c.start_line, c.end_line) for c in chunks[:3]
        ]
        return Answer(text=text, citations=citations)
