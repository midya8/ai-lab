"""Speculative sampling (Leviathan et al. 2023 / Chen et al. 2023).

Draft proposes k tokens; the target scores all positions; each proposal x is
accepted with prob min(1, p(x)/q(x)). On the first rejection, resample from
norm(max(p-q, 0)); if all k accepted, draw a bonus token from the target.
The output distribution is exactly the target's.
"""
import numpy as np


def _draw(p, rng):
    return int(rng.choice(len(p), p=p))


def autoregressive_generate(target, prompt, n, rng):
    out = list(prompt)
    for _ in range(n):
        out.append(_draw(target.probs(out), rng))
    return out[len(prompt):]


def speculative_generate(target, draft, prompt, n, k, rng, stats=None, controller=None):
    """Generate exactly n tokens. stats (dict) accumulates target_calls, proposed, accepted."""
    out = list(prompt)
    goal = len(prompt) + n
    st = stats if stats is not None else {}
    for key in ("target_calls", "proposed", "accepted"):
        st.setdefault(key, 0)
    while len(out) < goal:
        kcur = controller.k() if controller is not None else k
        kk = min(kcur, goal - len(out))
        ctx = list(out)
        qs, xs = [], []
        for _ in range(kk):
            q = draft.probs(ctx)
            x = _draw(q, rng)
            qs.append(q); xs.append(x); ctx.append(x)
        # one (batched, in a real system) target pass over kk+1 positions
        st["target_calls"] += 1
        st["proposed"] += kk
        ctx = list(out)
        rejected = False
        acc_before = st["accepted"]
        for q, x in zip(qs, xs):
            p = target.probs(ctx)
            if rng.random() < min(1.0, p[x] / q[x]):
                out.append(x); ctx.append(x); st["accepted"] += 1
            else:
                resid = np.maximum(p - q, 0.0)
                s = resid.sum()
                resid = resid / s if s > 0 else p
                out.append(_draw(resid, rng))
                rejected = True
                break
        if not rejected and len(out) < goal:
            out.append(_draw(target.probs(out), rng))
        if controller is not None:
            controller.update(st["accepted"] - acc_before, kk, rejected)
    return out[len(prompt):goal]


def theoretical_alpha(p, q):
    """Per-token acceptance probability sum_x min(p, q) for one context."""
    return float(np.minimum(p, q).sum())


def expected_speedup(alpha, k, cost_ratio):
    """Walltime improvement factor: (1-a^(k+1))/((1-a)(k*c+1)), c = draft/target cost."""
    if alpha >= 1.0:
        return (k + 1) / (k * cost_ratio + 1)
    return (1 - alpha ** (k + 1)) / ((1 - alpha) * (k * cost_ratio + 1))


class AdaptiveK:
    """Choose the draft length k each round from past rounds only.

    Because k never depends on the current round's random draws, the output
    distribution stays exactly the target's. Tracks an EMA of the per-token
    acceptance rate (MLE under a truncated-geometric model: accepted tokens /
    tokens tested, where a rejected round tested accepted+1 and a clean round
    tested k) and picks k maximising expected_speedup(alpha, k, cost_ratio).
    """

    def __init__(self, cost_ratio=0.1, k_min=1, k_max=16, k_init=4, decay=0.8):
        self.c, self.lo, self.hi, self.decay = cost_ratio, k_min, k_max, decay
        self.acc = 0.0
        self.tested = 0.0
        self._k = k_init

    def k(self):
        return self._k

    def alpha(self):
        return self.acc / self.tested if self.tested > 0 else None

    def update(self, accepted, proposed, rejected):
        tested = accepted + 1 if rejected else proposed
        self.acc = self.decay * self.acc + accepted
        self.tested = self.decay * self.tested + tested
        a = min(self.alpha(), 0.999)
        self._k = max(self.lo, min(self.hi, max(
            range(self.lo, self.hi + 1), key=lambda kk: expected_speedup(a, kk, self.c))))
