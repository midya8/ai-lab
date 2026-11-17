"""Train a small 2-layer transformer on the periodic-copy task with the microgrid autodiff engine."""
import numpy as np
from . import _mg  # noqa: F401  (puts microgrid on sys.path)
from microgrid import MiniGPT, softmax_cross_entropy, AdamW, clip_grad_norm
from .task import make_batch, predictable_mask, VOCAB, SEQ


def train_induction(seed=0, steps=4000, bs=32, lr=3e-3, d=32, heads=2, layers=2, log_every=500, verbose=False):
    """Returns (MiniGPT, history). The loss is plain next-token cross-entropy over all positions;
    metrics in the history are measured on the predictable positions only."""
    m = MiniGPT(VOCAB, d=d, heads=heads, layers=layers, ctx=SEQ - 1, seed=seed)
    opt = AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(1, steps + 1):
        x, P = make_batch(rng, bs)
        inp, tgt = x[:, :-1], x[:, 1:]
        opt.zero_grad()
        lg = m(inp)
        loss = softmax_cross_entropy(lg.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()
        clip_grad_norm(m.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == steps:
            mask = predictable_mask(P)
            l = lg.data
            z = l - l.max(-1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
            nll = -np.take_along_axis(lp, tgt[..., None], -1)[..., 0]
            hist.append((s, float(nll[mask].mean()), float((l.argmax(-1) == tgt)[mask].mean())))
            if verbose:
                print("step %d  predictable-position nll %.3f  acc %.3f" % hist[-1], flush=True)
    return m, hist
