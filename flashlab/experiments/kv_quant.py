"""KV-cache quantisation: attention error vs bytes per token. Writes experiments/kv_quant_results.txt.

SYNTHETIC data: keys have a few large-scale "outlier" channels (an assumption modelled on what
KV-quantisation papers report for LLM keys - not measured here, no real model is involved);
values are isotropic Gaussian. Errors are therefore indicative of the *mechanisms* (scale
granularity, K vs V sensitivity), not of any real model's perplexity.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flashlab import naive_attention, quant  # noqa: E402

out = []


def P(s=""):
    print(s)
    out.append(s)


H, N, D, NQ, SEEDS = 4, 1024, 64, 64, 5
OUTLIER_CH, OUTLIER_SCALE = 4, 8.0


def make(seed, peaky):
    r = np.random.default_rng(seed)
    ch = np.ones(D)
    ch[r.choice(D, OUTLIER_CH, replace=False)] = OUTLIER_SCALE
    k = r.normal(size=(H, N, D)) * ch
    v = r.normal(size=(H, N, D))
    q = r.normal(size=(H, NQ, D)) * (3.0 if peaky else 1.0)
    return q, k, v


def attn(q, k, v):
    # the last NQ queries of the sequence, causal
    return naive_attention(q, k, v, causal=True)[:2]


def roundtrip(x, bits, g):
    return quant.dequantize(*quant.quantize(x, bits, g), g)


def per_channel_roundtrip(k, bits, block=16):
    """KIVI-style: scale per channel across a block of `block` tokens (offline comparison only)."""
    kb = k.reshape(k.shape[0], -1, block, k.shape[-1]).transpose(0, 1, 3, 2)   # (H, nb, D, block)
    return quant.dequantize(*quant.quantize(kb, bits, None), None).transpose(0, 1, 3, 2).reshape(k.shape)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def evaluate(kfun, vfun, peaky):
    errs, lses = [], []
    for s in range(SEEDS):
        q, k, v = make(s, peaky)
        ref, l0 = attn(q, k, v)
        o, l1 = attn(q, kfun(k), vfun(v))
        errs.append(rel(o, ref))
        lses.append(np.abs(l1 - l0).max())
    return float(np.mean(errs)), float(np.max(errs)), float(np.mean(lses))


fp16 = 2 * 2 * D                      # bytes per (token, kv head), K+V at 2 bytes each
ident = lambda x: x                   # noqa: E731
P(f"Setup: {H} kv heads, d={D}, N={N} cached tokens, last {NQ} queries, causal, {SEEDS} seeds.")
P(f"Keys have {OUTLIER_CH} outlier channels x{OUTLIER_SCALE:g} (synthetic assumption). Baseline = fp16 K+V = {fp16} bytes per token-head.")
P("rel err = ||out - out_exact|| / ||out_exact|| (mean over seeds); lse err = max |lse - lse_exact|.")

for peaky in (False, True):
    P()
    P(f"{'PEAKY queries (x3)' if peaky else 'DIFFUSE queries (x1)'}")
    P(f"  {'K bits':>6} {'V bits':>6} {'group':>6} {'bytes/tok':>10} {'compress':>9} {'rel err':>9} {'worst':>9} {'lse err':>9}")
    for kb, vb in ((8, 8), (8, 4), (4, 8), (4, 4), (4, 2), (2, 4), (2, 2)):
        for g in (64, 16):
            by = quant.bytes_per_vector(D, kb, g) + quant.bytes_per_vector(D, vb, g)
            m, w, l = evaluate(lambda x: roundtrip(x, kb, g), lambda x: roundtrip(x, vb, g), peaky)
            P(f"  {kb:>6} {vb:>6} {g:>6} {by:>10} {fp16 / by:>8.2f}x {m:>9.4f} {w:>9.4f} {l:>9.4f}")

P()
P("Which of K and V is more sensitive? (4-bit, group 16, the other tensor exact; diffuse / peaky)")
for name, kf, vf in (("K only", lambda x: roundtrip(x, 4, 16), ident),
                     ("V only", ident, lambda x: roundtrip(x, 4, 16))):
    a = evaluate(kf, vf, False)[0]
    b = evaluate(kf, vf, True)[0]
    P(f"  {name}: rel err {a:.4f} / {b:.4f}")

P()
P("Key scale layout at equal scale overhead (8 bytes/token-head): per-token group 16 vs per-channel over 16-token blocks. V exact.")
P(f"  {'K bits':>6} {'per-token g16':>14} {'per-channel b16':>16}   (diffuse | peaky)")
for kb in (4, 2):
    t = [evaluate(lambda x: roundtrip(x, kb, 16), ident, p)[0] for p in (False, True)]
    c = [evaluate(lambda x: per_channel_roundtrip(x, kb), ident, p)[0] for p in (False, True)]
    P(f"  {kb:>6} {t[0]:>6.4f} | {t[1]:.4f} {c[0]:>8.4f} | {c[1]:.4f}")

with open(os.path.join(os.path.dirname(__file__), "kv_quant_results.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
