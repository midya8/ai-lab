"""Circuit analyses: attention scores, head ablation, direct logit attribution, activation
patching and weight-based composition scores."""
import numpy as np
from .task import predictable_mask


def _run(model, ids, hooks=None):
    cache = {}
    logits = model.forward(ids, hooks, cache)
    return logits, cache


def _logsoftmax(lg):
    z = lg - lg.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def loss_and_acc(model, tokens, periods, hooks=None):
    """Mean NLL and top-1 accuracy of next-token prediction on the predictable positions."""
    ids, tgt = tokens[:, :-1], tokens[:, 1:]
    lp = _logsoftmax(model.forward(ids, hooks))
    mask = predictable_mask(periods, tokens.shape[1])
    nll = -np.take_along_axis(lp, tgt[..., None], -1)[..., 0]
    return float(nll[mask].mean()), float((lp.argmax(-1) == tgt)[mask].mean())


# ------------------------------------------------------------------ attention-pattern scores
def previous_token_scores(cache, layers):
    """(layers, H): mean attention from position t to t-1 (t >= 1)."""
    out = []
    for l in range(layers):
        pat = cache[f"L{l}.pattern"]
        T = pat.shape[-1]
        idx = np.arange(1, T)
        out.append(pat[:, :, idx, idx - 1].mean(axis=(0, 2)))
    return np.array(out)


def induction_scores(cache, layers, periods):
    """(layers, H): mean attention from query position t >= p to key position t-p+1, i.e. to the token
    that followed the previous occurrence of the current token."""
    out = np.zeros((layers, cache["L0.pattern"].shape[1]))
    for l in range(layers):
        pat = cache[f"L{l}.pattern"]
        B, H, T, _ = pat.shape
        acc = np.zeros(H)
        n = 0
        for b in range(B):
            p = int(periods[b])
            ts = np.arange(p, T)
            if len(ts) == 0:
                continue
            acc += pat[b][:, ts, ts - p + 1].sum(axis=1)
            n += len(ts)
        out[l] = acc / max(n, 1)
    return out


# ------------------------------------------------------------------ ablation
def zero_ablate(layer, head):
    def fn(x):
        x = x.copy()
        x[:, head] = 0.0
        return x
    return {f"L{layer}.head_out": fn}


def mean_ablate(layer, head, ref_cache):
    """Replace one head's output by its mean over the reference batch at each position."""
    mean = ref_cache[f"L{layer}.head_out"][:, head].mean(axis=0)  # (T, D)

    def fn(x):
        x = x.copy()
        x[:, head] = mean[None, : x.shape[2]]
        return x
    return {f"L{layer}.head_out": fn}


def merge_hooks(*hs):
    """Combine hook dicts; hooks on the same point are applied in order."""
    out = {}
    for h in hs:
        for k, f in h.items():
            if k in out:
                prev = out[k]
                out[k] = (lambda a, b: (lambda x: b(a(x))))(prev, f)
            else:
                out[k] = f
    return out


def head_ablation_table(model, tokens, periods, mode="zero"):
    """Rows (layer, head, nll, acc, d_nll) for every head, plus the unablated baseline."""
    base_nll, base_acc = loss_and_acc(model, tokens, periods)
    _, ref = _run(model, tokens[:, :-1])
    rows = []
    for l in range(model.layers):
        for h in range(model.H):
            hk = zero_ablate(l, h) if mode == "zero" else mean_ablate(l, h, ref)
            nll, acc = loss_and_acc(model, tokens, periods, hk)
            rows.append(dict(layer=l, head=h, nll=nll, acc=acc, d_nll=nll - base_nll))
    return dict(baseline=dict(nll=base_nll, acc=base_acc), rows=rows)


# ------------------------------------------------------------------ MLP neuron ablation
def neuron_ablate(layer, neurons, ref_cache=None):
    """Zero (or, with ref_cache, replace by the batch-and-position mean) the given hidden neurons."""
    idx = np.atleast_1d(np.asarray(neurons, dtype=int))
    mean = None if ref_cache is None else ref_cache[f"L{layer}.mlp_hidden"].mean(axis=0)  # (T, F)

    def fn(x):
        x = x.copy()
        x[:, :, idx] = 0.0 if mean is None else mean[None, : x.shape[1]][:, :, idx]
        return x
    return {f"L{layer}.mlp_hidden": fn}


def mlp_ablation_table(model, tokens, periods, mode="zero"):
    """Whole-MLP ablation per layer: rows (layer, nll, acc, d_nll) plus the baseline."""
    base_nll, base_acc = loss_and_acc(model, tokens, periods)
    _, ref = _run(model, tokens[:, :-1])
    rows = []
    for l in range(model.layers):
        F = ref[f"L{l}.mlp_hidden"].shape[-1]
        hk = neuron_ablate(l, np.arange(F), ref if mode == "mean" else None)
        nll, acc = loss_and_acc(model, tokens, periods, hk)
        rows.append(dict(layer=l, nll=nll, acc=acc, d_nll=nll - base_nll))
    return dict(baseline=dict(nll=base_nll, acc=base_acc), rows=rows)


def neuron_ablation_table(model, tokens, periods, layer, mode="zero", top=10):
    """Single-neuron ablations in one layer, sorted by damage (d_nll). Also returns the
    concentration: share of the summed positive damage carried by the top-`top` neurons, and
    the damage of ablating those `top` together (single effects need not add up)."""
    base_nll, _ = loss_and_acc(model, tokens, periods)
    _, ref = _run(model, tokens[:, :-1])
    F = ref[f"L{layer}.mlp_hidden"].shape[-1]
    rc = ref if mode == "mean" else None
    d = np.array([loss_and_acc(model, tokens, periods, neuron_ablate(layer, [j], rc))[0] - base_nll
                  for j in range(F)])
    order = np.argsort(-d)
    top_idx = order[:top]
    pos = np.clip(d, 0, None).sum()
    joint = loss_and_acc(model, tokens, periods, neuron_ablate(layer, top_idx, rc))[0] - base_nll
    return dict(layer=layer, d_nll=d, order=order, top_share=float(np.clip(d[top_idx], 0, None).sum() / pos) if pos > 0 else 0.0,
                joint_top_d_nll=float(joint), sum_top_d_nll=float(d[top_idx].sum()))


# ------------------------------------------------------------------ direct logit attribution
def logit_attribution(model, tokens, periods):
    """Decompose the correct-token logit (centred over the vocabulary) into additive contributions of
    the embeddings, each attention head, each MLP and the constant biases, holding the final
    LayerNorm's scale fixed. Returns (dict name -> mean contribution over predictable positions, total).
    `sum(contribs.values()) == total` up to float error."""
    ids, tgt = tokens[:, :-1], tokens[:, 1:]
    logits, cache = _run(model, ids)
    resid = cache["resid_final"]
    g, bln = model.p["lnf"]
    Wu, bu = model.p["head"]
    std = np.sqrt(((resid - resid.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)

    def contrib(c):
        c = c - c.mean(-1, keepdims=True)
        lg = (c / std * g) @ Wu
        lg = lg - lg.mean(-1, keepdims=True)
        return np.take_along_axis(lg, tgt[..., None], -1)[..., 0]

    comps = {"embed": cache["embed"]}
    for l, blk in enumerate(model.p["blocks"]):
        ho = cache[f"L{l}.head_out"]
        for h in range(model.H):
            comps[f"L{l}H{h}"] = ho[:, h]
        comps[f"L{l}.attn_bias"] = np.broadcast_to(blk["proj"][1], ho[:, 0].shape)
        comps[f"L{l}.mlp"] = cache[f"L{l}.mlp_out"]
    mask = predictable_mask(periods, tokens.shape[1])
    out = {k: float(contrib(v)[mask].mean()) for k, v in comps.items()}
    const = (bln @ Wu + bu)
    const = np.take_along_axis(np.broadcast_to(const - const.mean(), logits.shape), tgt[..., None], -1)[..., 0]
    out["final_ln_bias+unembed_bias"] = float(const[mask].mean())
    lc = logits - logits.mean(-1, keepdims=True)
    total = float(np.take_along_axis(lc, tgt[..., None], -1)[..., 0][mask].mean())
    return out, total


# ------------------------------------------------------------------ activation patching
def patch_head_effects(model, clean, corrupt, periods):
    """For every head: run the corrupted batch with that head's output replaced by its output on the
    clean batch, and score how much of the clean run's log-prob of the *clean* target is recovered
    (0 = no recovery, 1 = full). Clean and corrupt share periods and positions, not tokens."""
    ids_c, ids_x = clean[:, :-1], corrupt[:, :-1]
    tgt = clean[:, 1:]
    mask = predictable_mask(periods, clean.shape[1])
    _, ccache = _run(model, ids_c)

    def lp_target(ids, hk=None):
        lp = _logsoftmax(model.forward(ids, hk))
        return float(np.take_along_axis(lp, tgt[..., None], -1)[..., 0][mask].mean())

    hi = lp_target(ids_c)
    lo = lp_target(ids_x)
    rows = []
    for l in range(model.layers):
        for h in range(model.H):
            clean_out = ccache[f"L{l}.head_out"][:, h]

            def fn(x, h=h, clean_out=clean_out):
                x = x.copy()
                x[:, h] = clean_out
                return x
            v = lp_target(ids_x, {f"L{l}.head_out": fn})
            rows.append(dict(layer=l, head=h, recovery=(v - lo) / (hi - lo)))
    return dict(clean_logprob=hi, corrupt_logprob=lo, rows=rows)


# ------------------------------------------------------------------ weights-only composition
def _ov(model, l, h):
    blk = model.p["blocks"][l]
    D, dh = model.D, model.dh
    Wv = blk["qkv"][0][:, 2 * D:][:, h * dh:(h + 1) * dh]
    Wo = blk["proj"][0][h * dh:(h + 1) * dh]
    return Wv @ Wo  # (D, D)


def _qk(model, l, h, which):
    blk = model.p["blocks"][l]
    D, dh = model.D, model.dh
    off = {"q": 0, "k": D, "v": 2 * D}[which]
    return blk["qkv"][0][:, off:][:, h * dh:(h + 1) * dh]  # (D, dh)


def composition_scores(model, kind="k", src_layer=0, dst_layer=1):
    """(H, H) matrix: ||W_OV(src) @ W_{q|k|v}(dst)||_F / (||W_OV(src)||_F ||W_{q|k|v}(dst)||_F),
    ignoring LayerNorm. Entry [i, j] is how strongly source head i's output can feed destination
    head j's queries/keys/values. Also returns the mean score for random matrices of the same
    shape as a reference."""
    S = np.zeros((model.H, model.H))
    for i in range(model.H):
        ov = _ov(model, src_layer, i)
        for j in range(model.H):
            w = _qk(model, dst_layer, j, kind)
            S[i, j] = np.linalg.norm(ov @ w) / (np.linalg.norm(ov) * np.linalg.norm(w) + 1e-12)
    rng = np.random.default_rng(0)
    ref = []
    for _ in range(200):
        a = rng.normal(size=(model.D, model.D))
        b = rng.normal(size=(model.D, model.dh))
        ref.append(np.linalg.norm(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return S, float(np.mean(ref))


# ------------------------------------------------------------------ attention profiles
def period_offset_profile(cache, layer, periods, offsets=range(-3, 4)):
    """(H, len(offsets)): mean attention from query t >= p to key t-p+o, for each offset o.
    o=0 is the previous occurrence of the *current* token (a duplicate-token head); o=1 is the token
    after it (an induction head); o=-1 is the token before it."""
    pat = cache[f"L{layer}.pattern"]
    B, H, T, _ = pat.shape
    offsets = list(offsets)
    out = np.zeros((H, len(offsets)))
    n = 0
    for b in range(B):
        p = int(periods[b])
        for t in range(p, T):
            for j, o in enumerate(offsets):
                k = t - p + o
                if 0 <= k <= t:
                    out[:, j] += pat[b, :, t, k]
            n += 1
    return out / max(n, 1)


def uniform_attention_baseline(periods, seq):
    """Induction/offset score a head attending uniformly to the visible context would get:
    mean of 1/(t+1) over the queried positions t >= p (t is the 0-based query position)."""
    vals = []
    for p in periods:
        for t in range(int(p), seq - 1):
            vals.append(1.0 / (t + 1))
    return float(np.mean(vals))


# ------------------------------------------------------------------ behavioural copy test
def copy_fidelity(model, tokens, periods, rng, hooks=None, vocab=None):
    """Causal test of copying. For each sequence pick a query position t in [p, 2p-2] (so exactly one
    earlier copy of the answer exists), overwrite that earlier copy (position j = t+1-p, the token the
    model should copy) with a token that does not occur in the sequence, and ask whether the model's
    top prediction at t follows the edit. The control edits a different earlier position instead.
    Returns dict(follow=..., control=..., n=...): the fraction of cases whose argmax at t equals the
    new token when the true source is edited vs. when another position is."""
    vocab = vocab or model.V
    T = tokens.shape[1] - 1
    follow = control = n = 0
    ids = tokens[:, :-1]
    for b in range(len(ids)):
        p = int(periods[b])
        lo, hi = p, min(2 * p - 2, T - 1)
        if hi < lo:
            continue
        t = int(rng.integers(lo, hi + 1))
        j = t + 1 - p
        unused = np.setdiff1d(np.arange(vocab), ids[b])
        if len(unused) == 0 or p < 3:
            continue
        new = int(rng.choice(unused))
        others = [k for k in range(p) if k != j]
        j2 = int(rng.choice(others))
        a = ids[b:b + 1].copy()
        a[0, j] = new
        c = ids[b:b + 1].copy()
        c[0, j2] = new
        follow += int(model.forward(a, hooks)[0, t].argmax() == new)
        control += int(model.forward(c, hooks)[0, t].argmax() == new)
        n += 1
    return dict(follow=follow / max(n, 1), control=control / max(n, 1), n=n)
