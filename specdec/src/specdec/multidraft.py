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


def tree_generate(target, draft, prompt, n, depth, m, rng, stats=None):
    """Speculative decoding with an m-ary draft tree, verified level by level.

    One round = one (batched, in a real system) target pass. Within a round we
    walk down from the current context: at each level m i.i.d. draft candidates
    are tested with recursive rejection (multi_draft_step), so the emitted token
    is exactly target-distributed given the context. If a candidate is accepted
    we descend into its subtree (up to `depth` levels); if all m are rejected
    the residual sample ends the round; if every level accepted, one bonus token
    is drawn from the target. m=1 reduces to ordinary speculative sampling.
    stats: target_calls, draft_nodes (m per level visited), accepted.
    """
    out = list(prompt)
    goal = len(prompt) + n
    st = stats if stats is not None else {}
    for key in ("target_calls", "draft_nodes", "accepted"):
        st.setdefault(key, 0)
    while len(out) < goal:
        st["target_calls"] += 1
        clean = True
        for _ in range(min(depth, goal - len(out))):
            st["draft_nodes"] += m
            tok, idx = multi_draft_step(target.probs(out), draft.probs(out), m, rng)
            out.append(tok)
            if idx is None:
                clean = False
                break
            st["accepted"] += 1
        if clean and len(out) < goal:
            out.append(int(rng.choice(len(target.probs(out)), p=target.probs(out))))
    return out[len(prompt):goal]


def tree_cost_per_token(stats, n_tokens, cost_ratio):
    """Simulated cost per generated token, in units of one target forward pass.

    cost = target_calls + cost_ratio * draft_nodes, where cost_ratio is the price
    of one draft forward relative to one target forward. Autoregressive decoding
    costs 1.0 per token, so values < 1 mean a net saving. Ignores batching
    overheads of verifying wide trees (a real system pays more as m grows).
    """
    return (stats["target_calls"] + cost_ratio * stats["draft_nodes"]) / n_tokens
