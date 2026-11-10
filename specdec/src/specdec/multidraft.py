"""Multi-draft speculative sampling at one position (recursive rejection sampling).

m candidates are drawn i.i.d. from the draft q. They are tested in order against
the target p: candidate x is accepted w.p. min(1, p(x)/q(x)); on rejection the
target is replaced by the normalised residual max(p-q, 0) and the next candidate
is tested. If all m are rejected, one token is drawn from the last residual.
The emitted token is distributed exactly as the original p (m=1 is the standard
rule). More candidates raise the chance that one is accepted.
"""
import numpy as np


def _residual(p, q):
    r = np.maximum(p - q, 0.0)
    s = r.sum()
    return r / s if s > 0 else p


def multi_draft_step(p, q, m, rng):
    """Return (token, index of accepted candidate or None)."""
    cur = p
    for i in range(m):
        x = int(rng.choice(len(q), p=q))
        if rng.random() < min(1.0, cur[x] / q[x]):
            return x, i
        cur = _residual(cur, q)
    return int(rng.choice(len(cur), p=cur)), None


def multi_draft_acceptance(p, q, m):
    """Exact P(some candidate accepted) = 1 - prod_i (1 - alpha_i) over the residual chain."""
    cur, rej = p, 1.0
    for _ in range(m):
        rej *= 1.0 - float(np.minimum(cur, q).sum())
        cur = _residual(cur, q)
    return 1.0 - rej


def multi_draft_generate(target, draft, prompt, n, m, rng, stats=None):
    """Generate n tokens, m draft candidates per position. stats: steps, accepted."""
    out = list(prompt)
    st = stats if stats is not None else {}
    st.setdefault("steps", 0); st.setdefault("accepted", 0)
    for _ in range(n):
        tok, idx = multi_draft_step(target.probs(out), draft.probs(out), m, rng)
        out.append(tok)
        st["steps"] += 1
        st["accepted"] += idx is not None
    return out[len(prompt):]
