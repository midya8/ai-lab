"""Corpus-grounded spelling correction for questions.

Lexical retrieval fails silently on a typo: "retrun" matches nothing, so
the question either abstains or answers from the wrong sentence.
``SpellCorrector`` repairs a question *only* against words that actually
occur in the indexed documents, so it can never invent vocabulary:

* a word already in the corpus (as written, or via its stem) is untouched;
* stopwords, numbers and words under 4 letters are never touched (4-letter
  words only when two adjacent letters are swapped, e.g. "opne" -> "open");
* a candidate must be within an optimal-string-alignment (Damerau-
  Levenshtein) distance of 1 (2 for words of 9+ letters);
* a tie between equally close candidates is broken by corpus frequency,
  then alphabetically, so results are deterministic;
* every change is reported, so the caller can show "did you mean".

The conservative bounds are deliberate: an unanswerable question such as
"capital of France" must still abstain, and a wrong "correction" toward a
document word would be a confidently wrong answer.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ragkit.retrieve import STOPWORDS, TOKEN_RE, tokenize

WORD_RE = re.compile(r"[A-Za-z]+")
MIN_LEN = 5
SHORT_LEN = 4


def osa_distance(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance, or limit + 1 once it must exceed ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] | None = None
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1] if prev[-1] <= limit else limit + 1


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    distance: int


class SpellCorrector:
    def __init__(self, texts):
        words: Counter[str] = Counter()
        for t in texts:
            for w in TOKEN_RE.findall(t.lower()):
                if w.isalpha() and w not in STOPWORDS:
                    words[w] += 1
        self.freq = words
        self.stems = {s for w in words for s in tokenize(w)}
        # candidate string -> (frequency, surface form shown to the user). Raw words
        # map to themselves; a stem ("return") is also a candidate because the
        # corpus may only contain inflections ("returns"), and shows the commonest one.
        self._cand: dict[str, tuple[int, str]] = {w: (n, w) for w, n in words.items()}
        for w, n in words.items():
            for st in tokenize(w):
                if st not in self._cand or (st not in words and n > self._cand[st][0]):
                    self._cand[st] = (n, w)
        self._by_len: dict[int, list[str]] = {}
        for c in self._cand:
            if len(c) >= SHORT_LEN:
                self._by_len.setdefault(len(c), []).append(c)

    def known(self, word: str) -> bool:
        w = word.lower()
        return w in self.freq or any(s in self.stems for s in tokenize(w))

    def suggest(self, word: str) -> Correction | None:
        w = word.lower()
        if len(w) < SHORT_LEN or w in STOPWORDS or self.known(w):
            return None
        limit = 2 if len(w) >= 9 else 1
        short = len(w) < MIN_LEN  # 4-letter words: only a swap of two adjacent letters ("opne")
        best: tuple[int, int, str] | None = None
        for n in range(len(w) - limit, len(w) + limit + 1):
            for cand in self._by_len.get(n, ()):
                d = osa_distance(w, cand, limit)
                if d > limit or (short and (n != len(w) or sorted(cand) != sorted(w))):
                    continue
                key = (d, -self._cand[cand][0], cand)
                if best is None or key < best:
                    best = key
        if best is None:
            return None
        return Correction(w, self._cand[best[2]][1], best[0])

    def correct(self, question: str) -> tuple[str, list[Correction]]:
        fixes: list[Correction] = []

        def sub(m: re.Match) -> str:
            c = self.suggest(m.group(0))
            if c is None:
                return m.group(0)
            fixes.append(c)
            w = m.group(0)
            return c.corrected.capitalize() if w[:1].isupper() else c.corrected

        return WORD_RE.sub(sub, question), fixes
