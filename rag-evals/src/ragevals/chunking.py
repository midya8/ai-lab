"""Sentence-aware sliding-window chunker."""
from __future__ import annotations
import re
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def chunk_text(doc_id: str, text: str, max_words: int = 60, overlap_sents: int = 1) -> list[Chunk]:
    """Pack whole sentences into chunks of <= max_words (a single longer sentence stays whole),
    repeating the last `overlap_sents` sentences at the start of the next chunk."""
    if max_words < 1 or overlap_sents < 0:
        raise ValueError("max_words must be >=1 and overlap_sents >=0")
    sents = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    chunks: list[Chunk] = []
    cur: list[str] = []
    fresh = 0  # sentences in cur not yet emitted in a chunk
    count = 0
    for s in sents:
        n = len(s.split())
        if cur and count + n > max_words:
            chunks.append(Chunk(doc_id, len(chunks), " ".join(cur)))
            cur = cur[-overlap_sents:] if overlap_sents else []
            fresh = 0
            count = sum(len(x.split()) for x in cur)
            if count + n > max_words:  # overlap would overflow; drop it
                cur, count = [], 0
        cur.append(s)
        fresh += 1
        count += n
    if cur and fresh:
        chunks.append(Chunk(doc_id, len(chunks), " ".join(cur)))
    return chunks
