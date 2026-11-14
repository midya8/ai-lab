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
