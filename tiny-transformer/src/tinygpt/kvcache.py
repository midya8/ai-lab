"""KV-cache incremental decoding for the NumPy GPT.

Prefill runs the prompt once and stores per-layer keys/values; each new token then
costs O(T) attention instead of re-running the whole sequence (O(T^2) per step).
"""
from __future__ import annotations

import numpy as np

from .model import GPT, ln_fwd, softmax, gelu_fwd


def _step(model: GPT, tok: int, pos: int, cache: list):
    """Process one token at position `pos`, appending to cache. Returns logits (V,)."""
    P, cfg = model.params, model.cfg
    H, C = cfg.n_head, cfg.n_embd
    D = C // H
    x = P["wte"][tok] + P["wpe"][pos]
    for i in range(cfg.n_layer):
        h, _ = ln_fwd(x, P[f"h{i}.ln1_g"], P[f"h{i}.ln1_b"])
        qkv = h @ P[f"h{i}.qkv_W"] + P[f"h{i}.qkv_b"]
        q, k, v = (t.reshape(H, D) for t in np.split(qkv, 3))
        cache[i][0].append(k)
        cache[i][1].append(v)
        K = np.stack(cache[i][0], 1)  # (H,T,D)
        V = np.stack(cache[i][1], 1)
        a = softmax(np.einsum("hd,htd->ht", q, K) / np.sqrt(D))
        y = np.einsum("ht,htd->hd", a, V).reshape(C)
        x = x + y @ P[f"h{i}.proj_W"] + P[f"h{i}.proj_b"]
        h, _ = ln_fwd(x, P[f"h{i}.ln2_g"], P[f"h{i}.ln2_b"])
        g, _ = gelu_fwd(h @ P[f"h{i}.fc_W"] + P[f"h{i}.fc_b"])
        x = x + g @ P[f"h{i}.out_W"] + P[f"h{i}.out_b"]
    h, _ = ln_fwd(x, P["lnf_g"], P["lnf_b"])
    return h @ (P["wte"].T if "head_W" not in P else P["head_W"]) + P["head_b"]


def _prefill(model: GPT, ctx):
    cache = [([], []) for _ in range(model.cfg.n_layer)]
    logits = None
    for pos, t in enumerate(ctx):
        logits = _step(model, int(t), pos, cache)
    return logits, cache


from .sampling import sample  # noqa: E402


def generate_cached(model: GPT, idx, max_new_tokens, temperature=1.0, top_k=None, seed=0, top_p=None, repetition_penalty=1.0):
    """Same sampling semantics and RNG stream as GPT.generate, but uses a KV cache.

    When the sequence outgrows block_size the window slides (positions are learned,
    so the cache must be rebuilt for the new window, exactly as GPT.generate sees it).
    """
    rng = np.random.default_rng(seed)
    out = [int(t) for t in np.array(idx).reshape(-1)]
    B = model.cfg.block_size
    logits, cache = _prefill(model, out[-B:])
    for n in range(max_new_tokens):
        lg = logits.astype(np.float64)
        nxt = sample(lg, rng, temperature, top_k, top_p, history=out, repetition_penalty=repetition_penalty)
        out.append(nxt)
        if n == max_new_tokens - 1:
            break
        if len(out) <= B:
            logits = _step(model, nxt, len(out) - 1, cache)
        else:
            logits, cache = _prefill(model, out[-B:])
    return np.array(out)
