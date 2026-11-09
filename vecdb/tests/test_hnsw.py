import numpy as np
import pytest

from vecdb import FlatIndex, HNSWIndex, clustered, queries_like, recall_at_k


@pytest.fixture(scope="module")
def world():
    X, _ = clustered(1200, 16, n_clusters=15, spread=0.7, seed=5)
    Q = queries_like(X, 40, seed=6)
    flat = FlatIndex(16)
    flat.add(X)
    gt, gd = flat.search(Q, 10)
    h = HNSWIndex(16, M=12, ef_construction=80, seed=1)
    h.add(X)
    return X, Q, gt, gd, h


def test_high_recall_and_ef_helps(world):
    X, Q, gt, _, h = world
    r_lo = recall_at_k(h.search(Q, 10, ef=10)[0], gt)
    r_hi = recall_at_k(h.search(Q, 10, ef=100)[0], gt)
    assert r_hi >= r_lo and r_hi >= 0.95
    assert h.last_stats["scanned"] < len(X)           # far fewer distance evals than brute force


def test_returned_distances_are_exact_and_sorted(world):
    X, Q, _, _, h = world
    ids, d = h.search(Q[:5], 10, ef=60)
    for q, i, di in zip(Q[:5], ids, d):
        assert np.all(np.diff(di) >= -1e-6)
        assert np.allclose(di, ((q - X[i]) ** 2).sum(1), atol=1e-3)


def test_graph_invariants(world):
    X, _, _, _, h = world
    st = h.degree_stats()
    assert st["max_degree_l0"] <= h.M0 and st["levels"] >= 2
    for lev, lv in enumerate(h.links):
        cap = h.M0 if lev == 0 else h.M
        for node, nb in lv.items():
            assert len(nb) <= cap and node not in nb and len(set(nb)) == len(nb)
            assert h.levels[node] >= lev
            assert all(h.levels[x] >= lev for x in nb)
    assert h.levels[h.entry] == h.max_level
    # layer 0 is connected from the entry point (otherwise some rows are unreachable)
    seen, stack = {h.entry}, [h.entry]
    while stack:
        for x in h.links[0][stack.pop()]:
            if x not in seen:
                seen.add(x); stack.append(x)
    assert len(seen) / len(X) > 0.99


def test_deterministic_given_seed():
    X, _ = clustered(300, 8, n_clusters=5, seed=0)
    a, b = HNSWIndex(8, seed=3), HNSWIndex(8, seed=3)
    a.add(X); b.add(X)
    assert a.links == b.links and a.entry == b.entry
    c = HNSWIndex(8, seed=4); c.add(X)
    assert c.links != a.links


def test_delete_never_returned_but_graph_still_routes(world):
    X, Q, gt, _, _ = world
    h = HNSWIndex(16, M=12, ef_construction=80, seed=1)
    h.add(X)
    kill = gt[:, 0]
    h.remove(np.unique(kill))
    ids, _ = h.search(Q, 10, ef=100)
    assert not set(ids.ravel()) & set(kill.tolist())
    assert (ids >= 0).all()


def test_filtered_search_returns_only_matches(world):
    X, Q, _, _, h = world
    mask = np.arange(len(X)) % 4 == 0                  # 25% selectivity -> graph walk
    ids, _ = h.search(Q, 10, ef=100, mask=mask)
    assert not h.last_stats["brute_force"]
    assert all(r % 4 == 0 for r in ids.ravel()) and (ids >= 0).all()
    flat = FlatIndex(16); flat.add(X)
    want, _ = flat.search(Q, 10, mask=mask)
    assert recall_at_k(ids, want) >= 0.9


def test_very_selective_filter_falls_back_to_exact_scan(world):
    X, Q, _, _, h = world
    mask = np.zeros(len(X), dtype=bool)
    mask[::100] = True                                  # ~1% of rows
    ids, d = h.search(Q, 5, mask=mask)
    assert h.last_stats["brute_force"]
    flat = FlatIndex(16); flat.add(X)
    want, wd = flat.search(Q, 5, mask=mask)
    assert np.array_equal(ids, want) and np.allclose(d, wd, atol=1e-3)
    none = h.search(Q[:1], 5, mask=np.zeros(len(X), dtype=bool))
    assert (none[0] == -1).all()


def test_empty_and_single_element():
    h = HNSWIndex(4)
    ids, d = h.search([[0, 0, 0, 0]], 3)
    assert (ids == -1).all()
    h.add([[1, 2, 3, 4]])
    ids, d = h.search([[1, 2, 3, 4]], 3)
    assert ids[0].tolist() == [0, -1, -1] and d[0, 0] == 0


def test_incremental_add_matches_one_shot_quality(world):
    X, Q, gt, _, _ = world
    h = HNSWIndex(16, M=12, ef_construction=80, seed=1)
    for s in range(0, len(X), 100):
        h.add(X[s:s + 100])
    assert recall_at_k(h.search(Q, 10, ef=100)[0], gt) >= 0.95


def test_cosine_and_ip_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    for metric in ("cosine", "ip"):
        h, f = HNSWIndex(8, metric, M=12, ef_construction=100), FlatIndex(8, metric)
        h.add(X); f.add(X)
        want, _ = f.search(X[:20], 5)
        got, _ = h.search(X[:20], 5, ef=100)
        assert recall_at_k(got, want) >= 0.9, metric


def test_param_validation():
    with pytest.raises(ValueError):
        HNSWIndex(4, M=1)
    with pytest.raises(ValueError):
        HNSWIndex(4, ef_construction=0)
