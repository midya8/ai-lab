"""Token healing: fix the prompt-boundary artefact of BPE.

A prompt that ends mid-word (e.g. "http:") is tokenised with a last token the
model would rarely have produced in training, because a longer merged token
(e.g. "://") exists. Healing backs up one token and constrains the first
generated token to those whose bytes start with the removed bytes.
"""


def heal_prompt(tok, text: str, n_back: int = 1):
    """Return (ids, prefix_bytes, allowed_ids).

    ids: prompt tokens with the last n_back removed (specials are never removed).
    prefix_bytes: bytes of the removed tokens; generation must begin with them.
    allowed_ids: every token id whose bytes start with prefix_bytes (the removed
    tokens themselves are always included, so greedy decoding can reproduce the
    original prompt).
    """
    ids = tok.encode(text)
    removed = []
    for _ in range(n_back):
        if not ids or ids[-1] not in tok.vocab or ids[-1] in tok._inv_special:
            break
        removed.insert(0, ids.pop())
    prefix = b"".join(tok.vocab[i] for i in removed)
    allowed = [i for i, b in tok.vocab.items() if b.startswith(prefix)] if prefix else []
    return ids, prefix, sorted(allowed)


def healed_continuation(tok, ids, prefix: bytes, generated):
    """Decode ids + generated tokens, checking the first generated token honours prefix.

    Raises ValueError when the generated tokens' bytes do not begin with prefix
    (a first token that is shorter than prefix is allowed only if the following
    tokens complete it).
    """
    got = b"".join(tok.vocab[i] for i in generated)
    if not got.startswith(prefix):
        raise ValueError("generation does not start with the healed prefix")
    return tok.decode(list(ids) + list(generated))
