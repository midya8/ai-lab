"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023) for tinygpt.

A cheap draft model proposes `k` tokens; the target model scores all of them in ONE forward
pass and each is accepted with prob min(1, p/q). On the first rejection a token is drawn from the
residual max(0, p-q) (renormalised); if all are accepted a bonus token comes from the target.
The output distribution is exactly the target's (tested on a toy pair). No KV cache is used, so
this demonstrates the algorithm and the acceptance statistics, not wall-clock speed.
"""
import numpy as np

from .sampling import softmax


def _probs(model, ctx, temperature):
    """Next-token distributions for every position of ctx: (T, V), float64."""
    lg = model.forward(np.asarray(ctx, dtype=np.int64)[None, -model.cfg.block_size:])[0][0].astype(np.float64)
    if temperature <= 0:
        out = np.zeros_like(lg)
        out[np.arange(len(lg)), lg.argmax(-1)] = 1.0
        return out
    z = lg / temperature
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def speculative_generate(target, draft, idx, max_new_tokens, k=4, temperature=1.0, seed=0):
    """Returns (tokens, stats). stats: target_calls, proposed, accepted, acceptance_rate,
    tokens_per_target_call. Requires matching vocabularies; contexts are cropped to block_size
    (keep k + prompt within it)."""
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("draft and target must share a vocabulary")
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(seed)
    seq = [int(t) for t in np.asarray(idx).reshape(-1)]
    n0 = len(seq)
    calls = proposed = accepted = 0
    while len(seq) - n0 < max_new_tokens:
        kk = min(k, max_new_tokens - (len(seq) - n0))
        ctx = list(seq)
        qs, props = [], []
        for _ in range(kk):  # draft proposes autoregressively
            q = _probs(draft, ctx, temperature)[-1]
            t = int(q.argmax()) if temperature <= 0 else int(rng.choice(len(q), p=q))
            qs.append(q); props.append(t); ctx.append(t)
        P = _probs(target, ctx, temperature)[-(kk + 1):]  # one target pass scores all kk + bonus
        calls += 1
        proposed += kk
        n_ok = 0
        for i, t in enumerate(props):
            p, q = P[i], qs[i]
            if rng.random() < min(1.0, p[t] / max(q[t], 1e-300)):
                seq.append(t); n_ok += 1
            else:
                r = np.maximum(p - q, 0.0)
                s = r.sum()
                r = r / s if s > 0 else p
                seq.append(int(r.argmax()) if temperature <= 0 else int(rng.choice(len(r), p=r)))
                break
        else:
            pb = P[kk]
            seq.append(int(pb.argmax()) if temperature <= 0 else int(rng.choice(len(pb), p=pb)))
        accepted += n_ok
    seq = seq[: n0 + max_new_tokens]
    stats = {"target_calls": calls, "proposed": proposed, "accepted": accepted,
             "acceptance_rate": accepted / max(proposed, 1),
             "tokens_per_target_call": max_new_tokens / max(calls, 1)}
    return np.array(seq, dtype=np.int64), stats


def expected_tokens_per_call(alpha, k):
    """Leviathan et al. closed form: expected tokens emitted per target call when each draft
    token is accepted independently with probability alpha: (1 - alpha^(k+1)) / (1 - alpha)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if k < 1:
        raise ValueError("k must be >= 1")
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def sweep_k(target, draft, prompt, ks=(1, 2, 4, 6), max_new_tokens=48, temperature=1.0, seeds=(0, 1, 2)):
    """Measure acceptance and tokens/target-call per k, next to the i.i.d.-acceptance prediction
    computed from the measured acceptance rate. Rows: dict(k, alpha, measured, predicted)."""
    rows = []
    for k in ks:
        accs, tpc = [], []
        for s in seeds:
            _, st = speculative_generate(target, draft, prompt, max_new_tokens, k=k,
                                         temperature=temperature, seed=s)
            accs.append(st["acceptance_rate"]); tpc.append(st["tokens_per_target_call"])
        a = float(np.mean(accs))
        rows.append({"k": k, "alpha": a, "measured": float(np.mean(tpc)),
                     "predicted": expected_tokens_per_call(a, k)})
    return rows
