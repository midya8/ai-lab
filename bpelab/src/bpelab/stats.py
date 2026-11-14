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
