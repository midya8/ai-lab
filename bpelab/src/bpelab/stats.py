"""Tokenizer-agnostic quality statistics (work with any object with encode/vocab_size)."""
import math
import re
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)


def token_stats(tok, text: str) -> dict:
    """Fertility (tokens/word), bytes/token, vocab utilisation and Renyi(alpha=2.5) efficiency.

    Renyi efficiency (Zouhar et al. 2023) = H_alpha(unigram token dist) / log(vocab_size);
    higher means token frequencies are more evenly spread.
    """
    ids = tok.encode(text)
    words = _WORD.findall(text)
    if not ids or not words:
        raise ValueError("text must contain at least one word and one token")
    counts = Counter(ids)
    n = len(ids)
    probs = [c / n for c in counts.values()]
    alpha = 2.5
    renyi = math.log(sum(p ** alpha for p in probs)) / (1 - alpha)
    return {
        "tokens": n,
        "fertility": n / len(words),
        "bytes_per_token": len(text.encode("utf-8")) / n,
        "vocab_utilisation": len(counts) / tok.vocab_size,
        "renyi_efficiency": renyi / math.log(tok.vocab_size),
    }


def dead_vocab(tok, text: str, top: int = 5) -> dict:
    """Which token ids never occur in `text` (e.g. held-out data).

    Returns the dead ids, the dead fraction, and the `top` most frequent ids.
    A high dead fraction on held-out text suggests the vocab is over-sized for
    the domain (or the training corpus was unrepresentative).
    """
    ids = tok.encode(text)
    if not ids:
        raise ValueError("text produced no tokens")
    counts = Counter(ids)
    dead = [i for i in range(tok.vocab_size) if i not in counts]
    return {
        "dead_ids": dead,
        "dead_fraction": len(dead) / tok.vocab_size,
        "top": counts.most_common(top),
    }


def compare_tokenizers(toks: dict, text: str) -> list:
    """Side-by-side `token_stats` for named tokenizers on the same text.

    Returns rows sorted by fertility (fewest tokens per word first), each a
    dict with a `name` key plus the token_stats fields.
    """
    if not toks:
        raise ValueError("need at least one tokenizer")
    rows = [dict(name=n, **token_stats(t, text)) for n, t in toks.items()]
    return sorted(rows, key=lambda r: (r["fertility"], r["name"]))


def roundtrip_failures(tok, texts, limit: int = 10) -> list:
    """Texts for which decode(encode(t)) != t (needs tok.decode). Returns up to `limit` failures."""
    bad = []
    for t in texts:
        if tok.decode(tok.encode(t)) != t:
            bad.append(t)
            if len(bad) >= limit:
                break
    return bad


def worst_words(tok, text: str, top: int = 5) -> list:
    """Words the tokenizer splits into the most tokens: [(word, n_tokens), ...], ties by word."""
    seen = {}
    for w in _WORD.findall(text):
        if w not in seen:
            seen[w] = len(tok.encode(w))
    return sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


def transfer_matrix(toks: dict, corpora: dict) -> dict:
    """Domain-transfer table: bytes/token of each tokenizer on each corpus.

    Returns {tokenizer_name: {corpus_name: bytes_per_token}}. A tokenizer trained
    on one domain usually compresses others worse; the gap (row max - diagonal) shows it.
    """
    if not toks or not corpora:
        raise ValueError("need at least one tokenizer and one corpus")
    out = {}
    for tn, t in toks.items():
        row = {}
        for cn, text in corpora.items():
            n = len(t.encode(text))
            row[cn] = len(text.encode("utf-8")) / n if n else 0.0
        out[tn] = row
    return out


def _boundaries(tok, word: str) -> set:
    """Internal split offsets (in bytes) of `word` under `tok`, via per-id decode.

    Exact for ASCII words; a token that ends mid-character decodes with a
    replacement char, so non-ASCII words are only approximate.
    """
    pos, cuts = 0, set()
    ids = tok.encode(word)
    for i in ids[:-1]:
        pos += len(tok.decode([i]).encode("utf-8"))
        cuts.add(pos)
    return cuts


def segmentation_agreement(tok_a, tok_b, text: str) -> dict:
    """How similarly two tokenizers split the same words.

    Returns `identical` (fraction of distinct words segmented identically,
    weighted by occurrence), and boundary precision/recall/F1 of `tok_b`'s
    internal split points against `tok_a`'s (words with no boundary in either
    contribute nothing to the boundary counts).
    """
    words = _WORD.findall(text)
    if not words:
        raise ValueError("text must contain at least one word")
    same = tp = fp = fn = 0
    for w in words:
        a, b = _boundaries(tok_a, w), _boundaries(tok_b, w)
        same += a == b
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"words": len(words), "identical": same / len(words),
            "precision": p, "recall": r, "f1": f1}
