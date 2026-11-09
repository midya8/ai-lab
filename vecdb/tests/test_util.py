import numpy as np
import pytest

from vecdb.util import Metric, kmeans, sqdist, topk_smallest


def test_sqdist_matches_direct():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(7, 5)).astype(np.float32), rng.normal(size=(9, 5)).astype(np.float32)
    direct = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    assert np.allclose(sqdist(A, B), direct, atol=1e-4)
    assert (sqdist(A, A) >= 0).all()


def test_metric_validation_and_cosine_normalises():
    with pytest.raises(ValueError):
        Metric("manhattan")
    m = Metric("cosine")
    X = m.prep([[3.0, 4.0]])
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    with pytest.raises(ValueError):
        m.prep([[0.0, 0.0]])
    with pytest.raises(ValueError):
        Metric("l2").prep([[np.nan, 1.0]])


def test_cosine_report_is_one_minus_cos():
    m = Metric("cosine")
    a, b = m.prep([[1.0, 0.0]]), m.prep([[1.0, 1.0]])
    d = m.report(m.dist(a, b))
    assert d[0, 0] == pytest.approx(1 - np.cos(np.pi / 4), abs=1e-6)


def test_topk_sorted_ties_by_index_and_padding():
    D = np.array([[3.0, 1.0, 1.0, 2.0], [np.inf, np.inf, 5.0, np.inf]], dtype=np.float32)
    idx, vals = topk_smallest(D, 3)
    assert idx[0].tolist() == [1, 2, 3]            # tie between 1 and 2 -> lower index first
    assert idx[1].tolist() == [2, -1, -1]          # inf columns are reported as missing
    assert np.isinf(vals[1, 1])
    idx, _ = topk_smallest(D, 6)                   # k > n pads
    assert idx.shape == (2, 6) and (idx[0, 4:] == -1).all()


def test_kmeans_recovers_separated_blobs_and_is_deterministic():
    rng = np.random.default_rng(1)
    true = np.array([[0, 0], [10, 10], [-10, 10]], dtype=np.float32)
    X = np.concatenate([c + rng.normal(scale=0.3, size=(100, 2)) for c in true]).astype(np.float32)
    c1, a1, inertia = kmeans(X, 3, seed=5)
    c2, a2, _ = kmeans(X, 3, seed=5)
    assert np.array_equal(a1, a2) and np.allclose(c1, c2)
    for t in true:
        assert np.min(np.linalg.norm(c1 - t, axis=1)) < 0.2
    assert inertia < 3 * 100 * 2 * 0.3 ** 2 * 1.5


def test_kmeans_handles_duplicates_and_too_few_points():
    X = np.zeros((20, 3), dtype=np.float32)
    c, a, inertia = kmeans(X, 4, seed=0)
    assert c.shape == (4, 3) and inertia == 0.0 and len(a) == 20
    with pytest.raises(ValueError):
        kmeans(X[:3], 4)
