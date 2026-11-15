"""Acquisition functions for MINIMISATION. All return 'higher is better' scores over candidate points."""
from __future__ import annotations
import math
import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(z):
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.asarray(_erf(z / math.sqrt(2.0)), dtype=float))


def norm_pdf(z):
    z = np.asarray(z, float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def expected_improvement(mu, sigma, best, xi=0.0):
    """EI(x) = E[max(best - xi - f(x), 0)] for f(x) ~ N(mu, sigma^2). Closed form; exactly 0 where sigma=0 and mu>=best-xi."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    imp = best - xi - mu
    out = np.maximum(imp, 0.0)
    ok = sigma > 1e-12
    z = np.where(ok, imp / np.where(ok, sigma, 1.0), 0.0)
    ei = imp * norm_cdf(z) + sigma * norm_pdf(z)
    return np.where(ok, np.maximum(ei, 0.0), out)


def probability_of_improvement(mu, sigma, best, xi=0.0):
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    imp = best - xi - mu
    ok = sigma > 1e-12
    z = np.where(ok, imp / np.where(ok, sigma, 1.0), 0.0)
    return np.where(ok, norm_cdf(z), (imp > 0).astype(float))


def confidence_bound(mu, sigma, beta=2.0):
    """Negated lower confidence bound, beta*sigma - mu (so that larger is better when minimising)."""
    return beta * np.asarray(sigma, float) - np.asarray(mu, float)


def mc_expected_improvement(samples, best):
    """Monte-Carlo EI from joint posterior samples (n_samples, n_points): mean of max(best - f, 0)."""
    return np.maximum(best - np.asarray(samples), 0.0).mean(0)


def mc_batch_expected_improvement(samples, best):
    """q-EI of a batch from joint samples (n_samples, q): E[max(best - min_j f_j, 0)]."""
    return np.maximum(best - np.asarray(samples).min(1), 0.0).mean()


def _log_h(z):
    """log h(z), h(z) = phi(z) + z*Phi(z) (so EI = sigma*h(z)), stable for very negative z.
    Direct for z > -10 (Phi via erfc, no 1+erf cancellation); for z <= -10 the asymptotic series
    h(z) = phi(z) * (1/t^2 - 3/t^4 + 15/t^6 - ...), t = -z, which avoids the catastrophic cancellation."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    big = z > -10.0
    zb = z[big]
    cdf = 0.5 * np.asarray(np.frompyfunc(math.erfc, 1, 1)(-zb / math.sqrt(2.0)), dtype=float)
    out[big] = np.log(np.maximum(norm_pdf(zb) + zb * cdf, 1e-300))
    zs = z[~big]
    t2 = zs * zs
    ser = np.zeros_like(zs)
    coef = 1.0
    for k in range(1, 13):
        ser += (-1.0) ** (k + 1) * coef / t2 ** k
        coef *= 2 * k + 1
    out[~big] = -0.5 * t2 - 0.5 * math.log(2.0 * math.pi) + np.log(ser)
    return out


def log_expected_improvement(mu, sigma, best, xi=0.0):
    """log EI, finite where plain EI underflows to 0 (far-from-incumbent candidates), so an optimiser can still rank
    them and follow gradients. -inf only where sigma = 0 and mu >= best - xi."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    imp = best - xi - mu
    ok = sigma > 1e-12
    sg = np.where(ok, sigma, 1.0)
    val = np.log(sg) + _log_h(np.where(ok, imp / sg, 0.0))
    with np.errstate(divide="ignore"):
        det = np.log(np.maximum(imp, 0.0))
    return np.where(ok, val, det)


def expected_improvement_grad(gp, Xs, best, xi=0.0):
    """EI and its analytic gradient w.r.t. the candidate points, for a fitted `GP` (minimisation).
    dEI/dx = -Phi(z) dmu/dx + phi(z) dsigma/dx. Returns (ei (n,), grad (n, d))."""
    mu, sd, dmu, dsd = gp.predict_grad(Xs)
    imp = best - xi - mu
    z = imp / sd
    ei = np.maximum(imp * norm_cdf(z) + sd * norm_pdf(z), 0.0)
    grad = -norm_cdf(z)[:, None] * dmu + norm_pdf(z)[:, None] * dsd
    return ei, grad
