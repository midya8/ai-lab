"""Numerical parity with tiny-transformer: copy a tinygpt.GPT's weights into MiniGPT.

Two independent from-scratch implementations (hand-written backprop vs. tape autodiff)
should agree on logits, loss and every parameter gradient when they share weights.
Supports the plain config (learned positions, MHA, untied head, no dropout).
"""
import os
import sys

import numpy as np


def _tinygpt():
    src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tiny-transformer", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import tinygpt.model as m
    return m


def load_from_tinygpt(model, gpt):
    """Copy gpt.params (float64 recommended) into the MiniGPT `model` in place."""
    P, cfg = gpt.params, gpt.cfg
    assert not cfg.rope and not cfg.tie_weights and cfg.kv_heads == cfg.n_head and cfg.dropout == 0
    assert len(model.blocks) == cfg.n_layer
    model.tok.data[...] = P["wte"]
    model.pos.data[...] = P["wpe"]
    for i, b in enumerate(model.blocks):
        g = lambda k: P[f"h{i}.{k}"]
        b.ln1.g.data[...], b.ln1.b.data[...] = g("ln1_g"), g("ln1_b")
        b.ln2.g.data[...], b.ln2.b.data[...] = g("ln2_g"), g("ln2_b")
        b.attn.qkv.w.data[...], b.attn.qkv.b.data[...] = g("qkv_W"), g("qkv_b")
        b.attn.proj.w.data[...], b.attn.proj.b.data[...] = g("proj_W"), g("proj_b")
        b.fc1.w.data[...], b.fc1.b.data[...] = g("fc_W"), g("fc_b")
        b.fc2.w.data[...], b.fc2.b.data[...] = g("out_W"), g("out_b")
    model.lnf.g.data[...], model.lnf.b.data[...] = P["lnf_g"], P["lnf_b"]
    model.head.w.data[...], model.head.b.data[...] = P["head_W"], P["head_b"]
    return model


def grads_by_tinygpt_name(model, n_layer):
    """MiniGPT parameter grads keyed by tinygpt's parameter names."""
    out = {"wte": model.tok.grad, "wpe": model.pos.grad, "lnf_g": model.lnf.g.grad,
           "lnf_b": model.lnf.b.grad, "head_W": model.head.w.grad, "head_b": model.head.b.grad}
    for i, b in enumerate(model.blocks):
        out.update({f"h{i}.ln1_g": b.ln1.g.grad, f"h{i}.ln1_b": b.ln1.b.grad,
                    f"h{i}.ln2_g": b.ln2.g.grad, f"h{i}.ln2_b": b.ln2.b.grad,
                    f"h{i}.qkv_W": b.attn.qkv.w.grad, f"h{i}.qkv_b": b.attn.qkv.b.grad,
                    f"h{i}.proj_W": b.attn.proj.w.grad, f"h{i}.proj_b": b.attn.proj.b.grad,
                    f"h{i}.fc_W": b.fc1.w.grad, f"h{i}.fc_b": b.fc1.b.grad,
                    f"h{i}.out_W": b.fc2.w.grad, f"h{i}.out_b": b.fc2.b.grad})
    return out


def compare(vocab=11, d=16, heads=4, layers=2, ctx=8, batch=3, seed=0):
    """Return dict of max abs differences: logits, loss, and the worst gradient (with its name)."""
    m = _tinygpt()
    from .attn import MiniGPT
    from .lm import lm_loss
    gpt = m.GPT(m.GPTConfig(vocab_size=vocab, block_size=ctx, n_layer=layers, n_head=heads, n_embd=d),
                seed=seed, dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    # tinygpt's init is tiny (std 0.02) and biases/LN are trivial; randomise so the test is not vacuous
    for k, v in gpt.params.items():
        v += rng.normal(0, 0.1, v.shape)
    x = rng.integers(0, vocab, (batch, ctx))
    y = rng.integers(0, vocab, (batch, ctx))
    logits, loss, cache = gpt.forward(x, y)
    _, grads = gpt.loss_and_grads(x, y)
    mg = load_from_tinygpt(MiniGPT(vocab, d, heads, layers, ctx, seed), gpt)
    for p in mg.parameters():
        p.grad = np.zeros_like(p.data)
    out = mg(x)
    l2 = lm_loss(mg, x, y)
    l2.backward()
    mgg = grads_by_tinygpt_name(mg, layers)
    diffs = {k: float(np.abs(mgg[k] - grads[k]).max()) for k in grads}
    worst = max(diffs, key=diffs.get)
    return {"logits": float(np.abs(out.data - logits).max()), "loss": abs(float(l2.data) - loss),
            "grad": diffs[worst], "worst_param": worst, "n_params": len(diffs)}
