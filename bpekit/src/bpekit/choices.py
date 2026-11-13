"""Token-level constrained decoding over a fixed set of string choices.

A byte-level BPE vocabulary can spell the same choice many ways (e.g. "positive"
as one merged token or as several). ``ChoiceConstraint`` tracks which choices
are still consistent with the bytes emitted so far and reports, at every step,
exactly the token ids that keep at least one choice reachable. Any tokenisation
path is allowed, not only the canonical one, so a model is never forced into an
unusual segmentation unless the vocabulary leaves no other option.
"""


class ChoiceConstraint:
    def __init__(self, tok, choices, eos_id: int | None = None):
        if not choices:
            raise ValueError("choices must be non-empty")
        self.tok = tok
        self.choices = [c.encode() if isinstance(c, str) else bytes(c) for c in choices]
        if any(not c for c in self.choices):
            raise ValueError("empty choice")
        self.eos_id = eos_id
        self.reset()

    def reset(self):
        self.emitted = b""
        self.ids = []

    def _alive(self):
        n = len(self.emitted)
        return [c for c in self.choices if c.startswith(self.emitted)] if n else list(self.choices)

    @property
    def done(self) -> bool:
        """True when emitted bytes exactly equal a choice that nothing longer extends."""
        alive = self._alive()
        return bool(self.emitted) and all(c == self.emitted for c in alive) and bool(alive)

    @property
    def complete(self) -> bool:
        """True when emitted bytes equal some choice (it may still be extendable)."""
        return self.emitted in self.choices

    def allowed(self):
        """Sorted token ids that keep at least one choice reachable.

        When the emitted bytes already equal a choice and eos_id is set, eos is allowed.
        """
        alive = self._alive()
        n = len(self.emitted)
        out = []
        for i, b in self.tok.vocab.items():
            if not b:
                continue
            for c in alive:
                rest = c[n:]
                if rest.startswith(b):
                    out.append(i)
                    break
        if self.eos_id is not None and self.complete:
            out.append(self.eos_id)
        return sorted(set(out))

    def push(self, token_id: int):
        if token_id not in self.allowed():
            raise ValueError(f"token {token_id} violates the constraint")
        if token_id == self.eos_id:
            return
        self.emitted += self.tok.vocab[token_id]
        self.ids.append(token_id)

    def result(self) -> str:
        if not self.complete:
            raise ValueError("no choice completed yet")
        return self.emitted.decode()


def constrained_pick(tok, choices, score_fn, eos_id=None, max_steps=64):
    """Greedy decode under a ChoiceConstraint. score_fn(ids_so_far, allowed) -> dict/seq of
    scores indexed by token id (higher wins). Stops when only eos or nothing extends."""
    cc = ChoiceConstraint(tok, choices, eos_id)
    for _ in range(max_steps):
        if cc.done:
            break
        allowed = cc.allowed()
        scores = score_fn(list(cc.ids), allowed)
        best = max(allowed, key=lambda i: scores[i])
        if best == eos_id:
            break
        cc.push(best)
    return cc.result(), cc.ids
