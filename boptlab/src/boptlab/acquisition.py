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
