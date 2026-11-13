"""Byte-level BPE tokenizer trainer (GPT-2 style), pure stdlib.

Base vocab = 256 bytes, so any string round-trips losslessly. Training uses
incremental pair-count updates (only words containing the merged pair are
touched) rather than recounting the corpus each merge.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

# Pre-tokenisation: words keep their leading space; merges never cross words.
PRETOKEN_RE = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+|.""", re.S)


def _pairs(word):
    return zip(word, word[1:])


def train_bpe(text: str, vocab_size: int, special_tokens=()):
    """Return (merges, special_tokens). merges is a list of (id_a, id_b); new id = 256 + index."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256")
    words = Counter(tuple(w.encode("utf-8")) for w in PRETOKEN_RE.findall(text))
    words = {w: c for w, c in words.items() if len(w) > 0}
    pair_counts = Counter()
    where = defaultdict(set)
    for w, c in words.items():
        for p in _pairs(w):
            pair_counts[p] += c
            where[p].add(w)
    merges = []
    n_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(max(0, n_merges)):
        if not pair_counts:
            break
        # deterministic: highest count, ties by smallest pair
        best = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        pair, cnt = best
        if cnt < 2:
            break
        new_id = 256 + len(merges)
        merges.append(pair)
        for w in list(where.get(pair, ())):
            c = words.pop(w, None)
            if c is None:
                continue
            for p in _pairs(w):
                pair_counts[p] -= c
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                where[p].discard(w)
            nw = _merge_word(w, pair, new_id)
            words[nw] = words.get(nw, 0) + c
            for p in _pairs(nw):
                pair_counts[p] += c
                where[p].add(nw)
        pair_counts.pop(pair, None)
        where.pop(pair, None)
    return merges, list(special_tokens)


def _merge_word(w, pair, new_id):
    out, i = [], 0
    while i < len(w):
        if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges, special_tokens=()):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.special_tokens = list(special_tokens)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_ids = {s: base + i for i, s in enumerate(self.special_tokens)}
        self._special_rev = {v: s for s, v in self.special_ids.items()}
        self._cache = {}
        if self.special_tokens:
            self._special_re = re.compile("(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")
        else:
            self._special_re = None

    @classmethod
    def train(cls, text, vocab_size, special_tokens=()):
        m, s = train_bpe(text, vocab_size, special_tokens)
        return cls(m, s)

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    def _encode_word(self, word_bytes: bytes):
        got = self._cache.get(word_bytes)
        if got is not None:
            return got
        ids = list(word_bytes)
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = list(_merge_word(tuple(ids), best, 256 + best_rank))
        self._cache[word_bytes] = ids
        return ids

    def _encode_word_dropout(self, word_bytes: bytes, dropout, rng):
        """BPE-dropout (Provilkov et al. 2020): each candidate merge is skipped with prob `dropout`."""
        ids = list(word_bytes)
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p)
                if r is None or rng.random() < dropout:
                    continue
                if best_rank is None or r < best_rank:
                    best, best_rank = p, r
            if best is None:
                break
            ids = list(_merge_word(tuple(ids), best, 256 + best_rank))
        return ids

    def encode(self, text: str, dropout: float = 0.0, rng=None):
        """Encode text. With dropout > 0, sample a stochastic segmentation (still decodes losslessly)."""
        if dropout > 0:
            import random
            rng = rng or random.Random()
            chunks = self._special_re.split(text) if self._special_re else [text]
            out = []
            for ch in chunks:
                if ch in self.special_ids:
                    out.append(self.special_ids[ch])
                    continue
                for w in PRETOKEN_RE.findall(ch):
                    out.extend(self._encode_word_dropout(w.encode("utf-8"), dropout, rng))
            return out
        chunks = self._special_re.split(text) if self._special_re else [text]
        out = []
        for ch in chunks:
            if ch in self.special_ids:
                out.append(self.special_ids[ch])
                continue
            for w in PRETOKEN_RE.findall(ch):
                out.extend(self._encode_word(w.encode("utf-8")))
        return out

    def decode(self, ids) -> str:
        buf = bytearray()
        for i in ids:
            if i in self._special_rev:
                buf += self._special_rev[i].encode("utf-8")
            else:
                buf += self.vocab[i]
        return buf.decode("utf-8", errors="replace")

    def compression(self, text: str) -> float:
        """Bytes per token."""
        n = len(self.encode(text))
        return len(text.encode("utf-8")) / max(1, n)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["merges"], d["special_tokens"])
