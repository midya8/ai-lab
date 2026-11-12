"""Round-trip verification: decode(encode(x)) == x on text and fuzzed input."""
import random

_ALPHABET = ("abc XYZ019 \n\t.,;'\"-_()é ß日本語☃😀\u200b\x00\r"
             "<|endoftext|>")


def fuzz_strings(n: int = 200, max_len: int = 40, seed: int = 0):
    rng = random.Random(seed)
    out = [""]
    for _ in range(n):
        out.append("".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, max_len))))
    return out


def verify(tok, texts=(), n_fuzz: int = 200, seed: int = 0):
    """Check lossless round trips (also with BPE-dropout). Returns a report.

    {"checked": int, "failures": [{"text": ..., "mode": ..., "got": ...}]}
    """
    failures, checked = [], 0
    rng = random.Random(seed)
    for t in list(texts) + fuzz_strings(n_fuzz, seed=seed):
        for mode, ids in (("plain", tok.encode(t)),
                          ("dropout", tok.encode(t, dropout=0.5, rng=random.Random(rng.random())))):
            checked += 1
            got = tok.decode(ids)
            if got != t:
                failures.append({"text": t, "mode": mode, "got": got})
    return {"checked": checked, "failures": failures}
