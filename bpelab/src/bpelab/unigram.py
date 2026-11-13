"""Unigram-LM tokenizer (Kudo 2018, SentencePiece style), pure stdlib.

Training: seed with all bytes + frequent word substrings, then alternate EM
(forward-backward over the segmentation lattice of each distinct word) and
pruning of the pieces whose removal costs the least corpus log-likelihood.
Encoding: Viterbi best segmentation. Byte pieces are never pruned, so every
string round-trips losslessly.
"""
from __future__ import annotations

import math
from collections import Counter

from .bpe import PRETOKEN_RE


def _words(text):
    c = Counter(w.encode("utf-8") for w in PRETOKEN_RE.findall(text))
    return {w: n for w, n in c.items() if w}


def _seed(words, size, max_len=8):
    sub = Counter()
    for w, n in words.items():
        for i in range(len(w)):
            for j in range(i + 2, min(len(w), i + max_len) + 1):
                sub[w[i:j]] += n
    return [p for p, _ in sub.most_common(size)]


def viterbi(word: bytes, logp: dict, max_len: int):
    n = len(word)
    best = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            lp = logp.get(word[i:j])
            if lp is not None and best[i] + lp > best[j]:
                best[j], back[j] = best[i] + lp, i
    out, j = [], n
    while j > 0:
        out.append(word[back[j]:j])
        j = back[j]
    return out[::-1], best[n]


def _em_counts(words, logp, max_len):
    """Expected piece counts and total log-likelihood via forward-backward."""
    exp, ll = Counter(), 0.0
    for w, cnt in words.items():
        n = len(w)
        fwd = [-math.inf] * (n + 1)
        fwd[0] = 0.0
        for j in range(1, n + 1):
            terms = [fwd[i] + logp[w[i:j]] for i in range(max(0, j - max_len), j) if w[i:j] in logp]
            if terms:
                m = max(terms)
                fwd[j] = m + math.log(sum(math.exp(t - m) for t in terms))
        bwd = [-math.inf] * (n + 1)
        bwd[n] = 0.0
        for i in range(n - 1, -1, -1):
            terms = [logp[w[i:j]] + bwd[j] for j in range(i + 1, min(n, i + max_len) + 1) if w[i:j] in logp]
            if terms:
                m = max(terms)
                bwd[i] = m + math.log(sum(math.exp(t - m) for t in terms))
        ll += cnt * fwd[n]
        for i in range(n):
            for j in range(i + 1, min(n, i + max_len) + 1):
                p = w[i:j]
                if p in logp:
                    exp[p] += cnt * math.exp(fwd[i] + logp[p] + bwd[j] - fwd[n])
    return exp, ll


def _normalise(exp, pieces):
    tot = sum(exp.get(p, 0.0) + 1e-9 for p in pieces)
    return {p: math.log((exp.get(p, 0.0) + 1e-9) / tot) for p in pieces}


class UnigramTokenizer:
    def __init__(self, pieces_logp: dict):
        self.logp = {bytes.fromhex(k) if isinstance(k, str) else k: v for k, v in pieces_logp.items()}
        for i in range(256):  # byte fallback guarantees losslessness
            self.logp.setdefault(bytes([i]), -30.0)
        self.max_len = max(len(p) for p in self.logp)
        self.pieces = sorted(self.logp, key=lambda p: (-self.logp[p], p))
        self.ids = {p: i for i, p in enumerate(self.pieces)}
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size, seed_size=4000, em_iters=3, shrink=0.8):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        words = _words(text)
        base = [bytes([i]) for i in range(256)]
        pieces = base + [p for p in _seed(words, seed_size) if len(p) > 1]
        max_len = max(len(p) for p in pieces)
        logp = _normalise(Counter({p: 1.0 for p in pieces}), pieces)
        while True:
            for _ in range(em_iters):
                exp, ll = _em_counts(words, logp, max_len)
                logp = _normalise(exp, list(logp))
            if len(logp) <= vocab_size:
                break
            # prune: loss of dropping a piece ~ its expected count * -logp (cheap proxy
            # for the exact re-segmentation loss used by SentencePiece)
            cand = sorted((p for p in logp if len(p) > 1), key=lambda p: exp.get(p, 0.0) * -logp[p])
            target = max(vocab_size, int(len(logp) * shrink))
            drop = set(cand[: len(logp) - target])
            if not drop:
                break
            logp = _normalise(exp, [p for p in logp if p not in drop])
        return cls(logp)

    @property
    def vocab_size(self):
        return len(self.pieces)

    def _encode_word(self, w: bytes):
        got = self._cache.get(w)
        if got is None:
            got = [self.ids[p] for p in viterbi(w, self.logp, self.max_len)[0]]
            self._cache[w] = got
        return got

    def encode(self, text: str):
        out = []
        for w in PRETOKEN_RE.findall(text):
            out.extend(self._encode_word(w.encode("utf-8")))
        return out

    def decode(self, ids) -> str:
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")

    def compression(self, text: str) -> float:
        return len(text.encode("utf-8")) / max(1, len(self.encode(text)))

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({k.hex(): v for k, v in self.logp.items()}, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            return cls(json.load(f))
