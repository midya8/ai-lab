import numpy as np
import pytest

from mamba import reverse_scan, scan, scan_parallel, scan_sequential


def brute(a, b, h0=None):
    B, T = b.shape[:2]
    h = np.zeros_like(b)
    prev = np.zeros_like(b[:, 0]) if h0 is None else h0
    for t in range(T):
        prev = a[:, t] * prev + b[:, t]
        h[:, t] = prev
    return h


@pytest.mark.parametrize("T", [1, 2, 3, 7, 8, 33])
def test_parallel_matches_sequential(T):
    rng = np.random.default_rng(T)
    a = rng.uniform(0.2, 1.1, (3, T, 4, 2))
    b = rng.normal(size=(3, T, 4, 2))
    h0 = rng.normal(size=(3, 4, 2))
    np.testing.assert_allclose(scan_parallel(a, b), scan_sequential(a, b), atol=1e-12)
    np.testing.assert_allclose(scan_parallel(a, b, h0), scan_sequential(a, b, h0), atol=1e-12)
    np.testing.assert_allclose(scan_sequential(a, b, h0), brute(a, b, h0), atol=1e-12)


def test_scan_matches_closed_form_for_constant_coefficient():
    a = np.full((1, 6, 1), 0.5)
    b = np.ones((1, 6, 1))
    h = scan(a, b)[0, :, 0]
    np.testing.assert_allclose(h, [2 * (1 - 0.5 ** (t + 1)) for t in range(6)])


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        scan(np.ones((1, 2)), np.ones((1, 2)), method="nope")


@pytest.mark.parametrize("method", ["sequential", "parallel"])
def test_reverse_scan_is_the_adjoint(method):
    """<w, scan(a, b)> differentiated w.r.t. b is reverse_scan(a, w): check by dot-product test."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.3, 1.0, (2, 9, 3))
    b = rng.normal(size=(2, 9, 3))
    w = rng.normal(size=(2, 9, 3))
    g = reverse_scan(a, w, method)
    # d/db_s of sum(w*h) via finite differences of the linear map
    eps = 1e-6
    num = np.zeros_like(b)
    for idx in np.ndindex(*b.shape):
        bp = b.copy(); bp[idx] += eps
        bm = b.copy(); bm[idx] -= eps
        num[idx] = ((w * scan(a, bp)).sum() - (w * scan(a, bm)).sum()) / (2 * eps)
    np.testing.assert_allclose(g, num, atol=1e-6)
