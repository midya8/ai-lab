"""Retrieval over chunks, implemented from scratch in the stdlib.

Two scorers are available:

* ``tfidf`` -- classic TF-IDF with cosine normalisation.
* ``bm25`` -- BM25 (k1=1.5, b=0.75), the default: it handles
  varying chunk lengths better, which matters for chunked docs.

Both add an exact-phrase boost: if the whole query (or a quoted
phrase inside it) appears verbatim in a chunk, that chunk scores
higher. Retrieval is deterministic: ties break on (source, start_line).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunk import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")
QUOTED_RE = re.compile(r'"([^"]+)"')


# Function words carry no topical signal and would otherwise let a
# query match a chunk on "a" or "the" alone -- in a small corpus
# such a match can score as highly as a real content-term match.
# They are dropped from documents and queries alike.
STOPWORDS = frozenset(
    """
    a an the is are was were be been being am
    do does did have has had
    i you he she it we they me him her us them
    my your his its our their mine yours ours theirs
    to of in on for with by at from into over under
    and or but not no nor so as if then than
    how what when where why who whom which that this these those
    there here all any some each every both either neither
    can could should would will shall may might must
    """.split()
)


def _stem(token: str) -> str:
    """Light suffix stripping so word variants match ("return" ~ "returns").

    Deliberately small: it only strips common English suffixes and
    applies identically to documents and queries, so matching stays
    consistent. It is not a full stemmer (e.g. "expire"/"expires"
    still differ) -- that is a documented limitation, not a bug.
    """
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 1 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 3 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    # Strip a trailing "e" so "close", "closed" and "closes" conflate.
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    # Stopwords are filtered on the raw token, before stemming.
    return [
        _stem(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS
    ]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def top_relative(scored: list[ScoredChunk], ratio: float = 0.5) -> list[ScoredChunk]:
    """Keep only chunks scoring at least *ratio* of the top score.

    Top-k retrieval always returns k chunks when it can, but the
    tail is often barely relevant. Quoting those chunks would add
    citations that do not support the answer, so answers are built
    from this filtered list instead. Recall is still measured on
    the full top-k list.
    """
    if not scored:
        return []
    cutoff = scored[0].score * ratio
    return [s for s in scored if s.score >= cutoff]


def select_for_answer(
    scored: list[ScoredChunk], query: str, ratio: float = 0.5
) -> list[ScoredChunk]:
    """Choose which retrieved chunks an answer may quote.

    The top chunk is always quoted. Any further chunk must also
    clear the relative-score bar *and* share at least two distinct
    content terms with the query: a chunk that matches on a single
    word ("bike" in a bike-shop corpus, "time" in a question about
    opening times) is usually on-topic but does not support the
    answer, and citing it would sink citation precision.
    """
    kept = top_relative(scored, ratio)
    if not kept:
        return []
    query_terms = set(tokenize(query))
    selected = [kept[0]]
    for item in kept[1:]:
        chunk_terms = set(tokenize(item.chunk.text))
        if len(query_terms & chunk_terms) >= 2:
            selected.append(item)
    return selected


class Retriever:
    """Rank chunks for a query. Build once, search many times."""

    def __init__(self, chunks: list[Chunk], scoring: str = "bm25"):
        if scoring not in ("tfidf", "bm25"):
            raise ValueError(f"unknown scoring: {scoring!r} (want 'tfidf' or 'bm25')")
        self.chunks = list(chunks)
        self.scoring = scoring
        self._doc_tokens = [tokenize(c.text) for c in self.chunks]
        self._doc_len = [len(tokens) for tokens in self._doc_tokens]
        self._avgdl = (
            sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        )
        doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            doc_freq.update(set(tokens))
        self._df = doc_freq
        n = len(self.chunks)
        # Smoothed IDF for TF-IDF; BM25's own IDF variant.
        self._idf_tfidf = {
            term: math.log((n + 1) / (df + 1)) + 1.0 for term, df in doc_freq.items()
        }
        self._idf_bm25 = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        # Precompute TF-IDF vectors and norms.
        self._tfidf_vectors: list[dict[str, float]] = []
        self._tfidf_norms: list[float] = []
        for tokens in self._doc_tokens:
            counts = Counter(tokens)
            vector = {
                term: (1.0 + math.log(count)) * self._idf_tfidf[term]
                for term, count in counts.items()
            }
            self._tfidf_vectors.append(vector)
            self._tfidf_norms.append(math.sqrt(sum(w * w for w in vector.values())))

    def _phrase_boost(self, query: str, chunk: Chunk) -> float:
        """+1.0 per distinct query phrase found verbatim in the chunk."""
        haystack = chunk.text.lower()
        phrases = set(QUOTED_RE.findall(query.lower()))
        plain = " ".join(query.lower().split())
        if plain:
            phrases.add(plain)
        return float(sum(1 for phrase in phrases if phrase and phrase in haystack))

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return the top-*k* chunks for *query*, best first.

        Returns an empty list for an empty query, an empty corpus,
        or when no chunk shares a term with the query. Scores are
        always > 0 for returned chunks.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []
        query_counts = Counter(tokens)
        scored: list[ScoredChunk] = []
        for idx, chunk in enumerate(self.chunks):
            if self.scoring == "tfidf":
                score = self._tfidf_score(idx, query_counts)
            else:
                score = self._bm25_score(idx, query_counts)
            if score > 0:
                score += self._phrase_boost(query, chunk)
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda s: (-s.score, s.chunk.source, s.chunk.start_line))
        return scored[:k]

    def _tfidf_score(self, idx: int, query_counts: Counter[str]) -> float:
        vector = self._tfidf_vectors[idx]
        norm = self._tfidf_norms[idx]
        if norm == 0:
            return 0.0
        dot = 0.0
        qnorm_sq = 0.0
        for term, count in query_counts.items():
            weight = (1.0 + math.log(count)) * self._idf_tfidf.get(term, 0.0)
            qnorm_sq += weight * weight
            dot += weight * vector.get(term, 0.0)
        if qnorm_sq == 0:
            return 0.0
        return dot / (norm * math.sqrt(qnorm_sq))

    def _bm25_score(self, idx: int, query_counts: Counter[str]) -> float:
        k1, b = 1.5, 0.75
        counts = Counter(self._doc_tokens[idx])
        dl = self._doc_len[idx]
        denom_base = k1 * (1.0 - b + b * (dl / self._avgdl)) if self._avgdl else k1
        score = 0.0
        for term in query_counts:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf_bm25.get(term, 0.0)
            score += idf * (freq * (k1 + 1.0)) / (freq + denom_base)
        return score
