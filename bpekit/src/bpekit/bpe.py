"""Byte-level BPE tokenizer trained from scratch (GPT-2 style, no regex pre-split lib)."""
from __future__ import annotations
import json, re
from collections import Counter

_SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def _merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def train(text: str, vocab_size: int = 300, specials=("<|endoftext|>",)) -> "BPETokenizer":
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256")
    words = Counter(_SPLIT.findall(text))
    seqs = {w: list(w.encode("utf-8")) for w in words}
    merges = []
    for n in range(256, vocab_size):
        stats = Counter()
        for w, s in seqs.items():
            for p in zip(s, s[1:]):
                stats[p] += words[w]
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))  # deterministic ties
        if stats[best] < 2:
            break
        merges.append(best)
        for w in seqs:
            seqs[w] = _merge(seqs[w], best, n)
    return BPETokenizer(merges, specials)


def train_fast(text: str, vocab_size: int = 300, specials=("<​|endoftext|>",)) -> "BPETokenizer":
    """Same merges as :func:`train`, but pair counts are updated incrementally.

    Keeps a pair -> {words containing it} index, so each merge only touches the
    words that actually contain the winning pair instead of rescanning the corpus.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256")
    words = Counter(_SPLIT.findall(text))
    seqs = {w: list(w.encode("utf-8")) for w in words}
    stats, where = Counter(), {}
    for w, s in seqs.items():
        for p in zip(s, s[1:]):
            stats[p] += words[w]
            where.setdefault(p, set()).add(w)
    merges = []
    for n in range(256, vocab_size):
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        if stats[best] < 2:
            break
        merges.append(best)
        for w in list(where.get(best, ())):
            s, c = seqs[w], words[w]
            for p in zip(s, s[1:]):
                stats[p] -= c
                if stats[p] <= 0:
                    del stats[p]
            s = _merge(s, best, n)
            seqs[w] = s
            for p in zip(s, s[1:]):
                stats[p] += c
                where.setdefault(p, set()).add(w)
        where.pop(best, None)
        stats.pop(best, None)
    return BPETokenizer(merges, specials)


class BPETokenizer:
    def __init__(self, merges, specials=()):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.specials = {s: base + i for i, s in enumerate(specials)}
        self._inv_special = {v: k for k, v in self.specials.items()}
        self._cache = {}

    def __len__(self):
        return 256 + len(self.merges) + len(self.specials)

    def _encode_word(self, w):
        if w in self._cache:
            return self._cache[w]
        ids = list(w.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = _merge(ids, pair, 256 + self.rank[pair])
        self._cache[w] = ids
        return ids

    def encode(self, text: str):
        if not self.specials:
            parts = [text]
        else:
            pat = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
            parts = re.split(pat, text)
        out = []
        for part in parts:
            if part in self.specials:
                out.append(self.specials[part])
            else:
                for w in _SPLIT.findall(part):
                    out.extend(self._encode_word(w))
        return out

    def decode(self, ids) -> str:
        b = b"".join(self._inv_special[i].encode() if i in self._inv_special else self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def compression(self, text: str) -> float:
        """bytes per token."""
        return len(text.encode("utf-8")) / max(1, len(self.encode(text)))

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges, "specials": list(self.specials)}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["merges"], d["specials"])


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable-unicode-char map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def export_gpt2(tok: BPETokenizer, vocab_path, merges_path):
    """Write GPT-2/HuggingFace-style vocab.json + merges.txt.

    Token strings use the GPT-2 byte-to-unicode alphabet. Only the merge table is
    portable: this package's pre-split regex is simpler than GPT-2's, so ids from
    a library using its own regex can differ on contractions and some punctuation.
    """
    b2u = bytes_to_unicode()
    s = lambda i: "".join(b2u[b] for b in tok.vocab[i])
    vocab = {s(i): i for i in range(256 + len(tok.merges))}
    for name, i in tok.specials.items():
        vocab[name] = i
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in tok.merges:
            f.write(f"{s(a)} {s(b)}\n")


def import_gpt2(vocab_path, merges_path, specials=()):
    """Rebuild a BPETokenizer from vocab.json + merges.txt written by export_gpt2
    (or any byte-level BPE with the GPT-2 alphabet and merge ids in rank order)."""
    u2b = {c: b for b, c in bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    to_id = lambda t: vocab[t]
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            merges.append((to_id(a), to_id(b)))
    for n, (a, b) in enumerate(merges):
        if a >= 256 + n or b >= 256 + n:
            raise ValueError(f"merge {n} uses a token id defined later; vocab does not match merge order")
    tok = BPETokenizer(merges, specials)
    for t, i in vocab.items():
        if i < len(tok.vocab) and t not in specials:
            if bytes(u2b[c] for c in t) != tok.vocab[i]:
                raise ValueError(f"vocab id {i} does not match merge order")
    return tok
