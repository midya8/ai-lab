"""Conformal test martingale for detecting exchangeability violations (Vovk et al.).

Online randomised conformal p-values are i.i.d. Uniform(0,1) iff the stream is
exchangeable; the simple-mixture power martingale then is a nonnegative martingale
with M_0 = 1, so by Ville's inequality P(sup_t M_t >= 1/delta) <= delta."""
import numpy as np


def online_conformal_pvalues(scores, rng):
    """Randomised p_t = (#{i<t: s_i > s_t} + U*(#{i<=t: s_i == s_t})) / t for a score stream
    (larger score = more non-conforming)."""
    s = np.asarray(scores, float)
    p = np.empty(len(s))
    for t in range(len(s)):
        past = s[:t + 1]
        gt = np.sum(past > s[t])
        eq = np.sum(past == s[t])
        p[t] = (gt + rng.random() * eq) / (t + 1)
    return p


def power_martingale_log(pvalues, epsilons=None):
    """log of the simple-mixture martingale: average over epsilons of prod eps * p^(eps-1).
    Returns the log-martingale path (length T)."""
    p = np.clip(np.asarray(pvalues, float), 1e-12, 1.0)
    eps = np.linspace(0.05, 0.95, 19) if epsilons is None else np.asarray(epsilons, float)
    # log of each component's running product, shape (T, E)
    comp = np.cumsum(np.log(eps)[None, :] + (eps - 1)[None, :] * np.log(p)[:, None], axis=0)
    m = comp.max(axis=1)
    return m + np.log(np.mean(np.exp(comp - m[:, None]), axis=1))


def detect_change(scores, delta=0.01, rng=None, epsilons=None):
    """Return (alarm_index or None, log-martingale path). Alarm at the first t with M_t >= 1/delta;
    false-alarm probability <= delta under exchangeability."""
    rng = np.random.default_rng(0) if rng is None else rng
    lm = power_martingale_log(online_conformal_pvalues(scores, rng), epsilons)
    hit = np.nonzero(lm >= np.log(1.0 / delta))[0]
    return (int(hit[0]) if len(hit) else None), lm
