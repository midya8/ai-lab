"""The synthetic task: periodic sequences of distinct tokens.

Each sequence repeats a random block of `p` distinct tokens (p drawn per sequence from
[pmin, pmax]). The first `p` tokens are unpredictable; after that the next token is fully
determined by the earlier text. Because `p` varies, a model cannot solve this with a fixed
positional offset ("copy the token H steps back"); it has to look the current token up
earlier in the context and copy what followed it: an induction mechanism.
"""
import numpy as np

VOCAB = 24
SEQ = 17  # tokens per sequence; the model sees SEQ-1 inputs and predicts the shifted targets


def make_batch(rng, bs, vocab=VOCAB, seq=SEQ, pmin=4, pmax=8):
    """Return (tokens (bs, seq), periods (bs,))."""
    assert pmax <= vocab and pmin >= 1
    out = np.zeros((bs, seq), dtype=int)
    periods = np.zeros(bs, dtype=int)
    for i in range(bs):
        p = int(rng.integers(pmin, pmax + 1))
        base = rng.permutation(vocab)[:p]
        out[i] = np.tile(base, seq // p + 1)[:seq]
        periods[i] = p
    return out, periods


def predictable_mask(periods, seq=SEQ):
    """mask[b, j] is True when input position j (predicting token j+1) is determined by context."""
    return (np.arange(seq - 1)[None, :] + 1) >= np.asarray(periods)[:, None]


def corrupt_batch(rng, tokens, periods, vocab=VOCAB):
    """A new batch with the same periods but fresh random blocks (for activation patching)."""
    seq = tokens.shape[1]
    out = np.zeros_like(tokens)
    for i, p in enumerate(periods):
        base = rng.permutation(vocab)[:p]
        out[i] = np.tile(base, seq // p + 1)[:seq]
    return out
