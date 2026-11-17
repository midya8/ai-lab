"""What does the step size ``delta`` learn? Probes for selective SSMs.

``delta_traces`` reads the per-token step size of every layer (averaged over channels). In a
selective-copying model a large ``delta`` means "write this token into the state", a ~0 ``delta``
means "skip it" (Abar -> 1, Bbar -> 0). ``gate_auc`` scores how well delta separates data tokens from
noise, and ``skip_ablation`` tests the causal claim by forcing delta to 0 at chosen positions.
"""
from __future__ import annotations

import contextlib

import numpy as np


def delta_traces(model, tokens):
    """Forward ``tokens`` and return a list (one per selective layer) of (B, T) channel-mean deltas."""
    model.forward(tokens)
    out = []
    for b in model.blocks:
        if b.ssm.selective:
            out.append(b.ssm.cache[2].mean(-1))
    return out


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), by ranks (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gate_auc(traces, tokens, seq_len):
    """Per layer: mean delta on data vs noise tokens (first ``seq_len`` positions) and the AUC of delta as a data detector."""
    data = tokens[:, :seq_len] > 0
    rows = []
    for d in traces:
        d = d[:, :seq_len]
        rows.append({"delta_data": float(d[data].mean()), "delta_noise": float(d[~data].mean()),
                     "auc": auc(d[data], d[~data])})
    return rows


@contextlib.contextmanager
def force_delta_zero(model, mask, layers=None):
    """Within the block, delta is 0 where ``mask`` (B, T) is True, in the chosen layers (default all):
    those tokens are neither written nor decay the state. Applies to the SSM layers only."""
    ssms = [b.ssm for b in model.blocks]
    chosen = range(len(ssms)) if layers is None else layers
    saved = []
    for k in chosen:
        s = ssms[k]
        orig = s._coeffs

        def patched(u, _orig=orig):
            z, delta, A, Abar, Bf, Cf = _orig(u)
            delta = np.where(mask[..., None], 0.0, delta)
            return z, delta, A, np.exp(delta[..., None] * A), Bf, Cf

        s._coeffs = patched
        saved.append((s, orig))
    try:
        yield
    finally:
        for s, orig in saved:
            s._coeffs = orig


def skip_ablation(model, tokens, targets, seq_len, rng):
    """Token accuracy when delta is zeroed at (a) nothing, (b) the data tokens, (c) as many random noise tokens."""
    from .tasks import score
    B, T = tokens.shape
    data = np.zeros((B, T), bool)
    data[:, :seq_len] = tokens[:, :seq_len] > 0
    noise = np.zeros((B, T), bool)
    for i in range(B):
        idx = np.flatnonzero(~data[i, :seq_len])
        noise[i, rng.choice(idx, data[i].sum(), replace=False)] = True
    res = {"intact": score(model.forward(tokens), targets)[0]}
    with force_delta_zero(model, data):
        res["skip_data"] = score(model.forward(tokens), targets)[0]
    with force_delta_zero(model, noise):
        res["skip_random_noise"] = score(model.forward(tokens), targets)[0]
    return res
