"""Sentence-aware chunking for ragkit.

Splits documents into overlapping chunks without cutting mid-sentence
where avoidable. Each chunk remembers its source file, line range and
the most recent Markdown heading above it, so answers can cite it.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'#>\-])")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    source: str  # path relative to the indexed docs directory
    text: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    heading: str = ""

    def citation(self) -> str:
        return f"[{self.source}:{self.start_line}-{self.end_line}]"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "heading": self.heading,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            source=data["source"],
            text=data["text"],
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            heading=data.get("heading", ""),
        )


def _line_starts(text: str) -> list[int]:
    """Character offset at which each line starts (line 1 starts at 0)."""
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def _line_of(line_starts: list[int], offset: int) -> int:
    """1-based line number containing the given character offset."""
    return bisect.bisect_right(line_starts, offset)


def _clean_sentence(part: str) -> str:
    """Strip Markdown heading/blockquote markers and collapse whitespace.

    The cleaned text is what gets indexed and quoted; offsets still
    refer to the original text, so line numbers stay correct.
    """
    part = re.sub(r"(?m)^#{1,6}\s+", "", part)
    part = re.sub(r"(?m)^>\s?", "", part)
    return re.sub(r"\s+", " ", part).strip()


def _sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into (sentence, start_offset, end_offset) triples.

    Text is first split into paragraphs (blank-line separated).
    Paragraphs that are only a Markdown heading or only a blockquote
    are skipped: headings are already recorded on the chunk as
    context, and blockquotes in these docs are boilerplate
    (e.g. the demo-data disclaimer) that would otherwise pollute
    every chunk and every quoted excerpt. Within a paragraph,
    splitting happens on whitespace after sentence-ending
    punctuation, so a chunk boundary never lands mid-sentence
    unless a single sentence is longer than the chunk size.
    """
    sentences: list[tuple[str, int, int]] = []
    pos = 0
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph.strip():
            continue
        start = text.find(paragraph, pos)
        if start == -1:  # pragma: no cover - defensive
            start = pos
        end = start + len(paragraph)
        pos = end
        lines = [line for line in paragraph.splitlines() if line.strip()]
        if lines and all(
            re.match(r"^\s*(#{1,6}\s+|>\s?)", line) for line in lines
        ):
            continue
        for part in SENTENCE_SPLIT_RE.split(paragraph):
            stripped = _clean_sentence(part)
            if not stripped:
                continue
            part_start = text.find(part, start)
            if part_start == -1:  # pragma: no cover - defensive
                part_start = start
            sentences.append((stripped, part_start, part_start + len(part)))
    return sentences


def split_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Split *text* into overlapping, sentence-aware chunks.

    chunk_size and overlap are measured in characters. A new chunk
    starts with the trailing sentences of the previous chunk (up to
    *overlap* characters) so context straddling a boundary is kept.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    if not text.strip():
        return []

    line_starts = _line_starts(text)
    headings = [(m.start(), m.group(2).strip()) for m in HEADING_RE.finditer(text)]

    def heading_at(offset: int) -> str:
        heading = ""
        for pos, title in headings:
            if pos <= offset:
                heading = title
            else:
                break
        return heading

    def flush(chunk_sentences: list[tuple[str, int, int]]) -> Chunk:
        joined = " ".join(s for s, _, _ in chunk_sentences)
        start = chunk_sentences[0][1]
        end = chunk_sentences[-1][2]
        return Chunk(
            source=source,
            text=joined,
            start_line=_line_of(line_starts, start),
            end_line=_line_of(line_starts, max(end - 1, start)),
            heading=heading_at(start),
        )

    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []

    def current_len() -> int:
        return len(" ".join(s for s, _, _ in current))

    for sentence in _sentences(text):
        if current and current_len() + 1 + len(sentence[0]) > chunk_size:
            chunks.append(flush(current))
            # Carry trailing sentences forward as overlap.
            tail: list[tuple[str, int, int]] = []
            tail_len = 0
            for prev in reversed(current):
                added = len(prev[0]) + (1 if tail else 0)
                if tail_len + added > overlap:
                    break
                tail.append(prev)
                tail_len += added
            current = list(reversed(tail))
        current.append(sentence)

    if current:
        chunks.append(flush(current))
    return chunks
