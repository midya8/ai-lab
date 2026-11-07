"""Shared next-token sampling: temperature, top-k, top-p (nucleus) and repetition penalty."""
import numpy as np


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def filter_logits(logits, top_k=None, top_p=None):
    """Mask logits outside the top-k and/or the smallest nucleus with mass >= top_p.

    The highest-probability token is always kept.
    """
    lg = np.array(logits, dtype=np.float64)
    if top_k:
        k = min(int(top_k), len(lg))
        lg[lg < np.sort(lg)[-k]] = -np.inf
    if top_p is not None and 0 < top_p < 1:
        order = np.argsort(-lg, kind="stable")
        probs = softmax(lg)[order]
        cum = np.cumsum(probs)
        # keep tokens until cumulative mass first reaches top_p
        cut = int(np.searchsorted(cum, top_p, side="left")) + 1
        lg[order[cut:]] = -np.inf
    return lg


def apply_repetition_penalty(logits, history, penalty=1.0):
    """CTRL-style penalty: logits of already-seen tokens are divided by `penalty`
    when positive and multiplied by it when negative (penalty > 1 discourages repeats)."""
    lg = np.array(logits, dtype=np.float64)
    if penalty is None or penalty == 1.0 or not len(history):
        return lg
    if penalty <= 0:
        raise ValueError("repetition penalty must be > 0")
    seen = np.unique(np.asarray(list(history), dtype=np.int64))
    seen = seen[(seen >= 0) & (seen < len(lg))]
    v = lg[seen]
    lg[seen] = np.where(v > 0, v / penalty, v * penalty)
    return lg


def sample(logits, rng, temperature=1.0, top_k=None, top_p=None, history=None, repetition_penalty=1.0):
    lg = np.asarray(logits, dtype=np.float64)
    if history is not None:
        lg = apply_repetition_penalty(lg, history, repetition_penalty)
    if temperature <= 0:
        return int(lg.argmax())
    lg = filter_logits(lg / temperature, top_k, top_p)
    return int(rng.choice(len(lg), p=softmax(lg)))
