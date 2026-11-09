import numpy as np
import pytest

from vecdb import Collection, FlatIndex, HNSWIndex, IVFIndex, clustered, load_index, save_index


def build(kind, n=400, **params):
    X, lab = clustered(n, 12, n_clusters=8, spread=0.6, seed=2)
    c = Collection(12, "cosine", kind, **params)
    if kind == "ivf":
        c.index.nlist = 8
    ids = [f"d{i}" for i in range(n)]
    metas = [{"topic": int(l), "year": 2000 + i % 25} for i, l in enumerate(lab)]
    c.upsert(ids, X, metas)
    if kind == "ivf":
        c.train()
    return c, X, ids, metas


@pytest.mark.parametrize("kind,params", [("flat", {}), ("hnsw", {"M": 8, "ef_construction": 60}), ("ivf", {"nlist": 8})])
def test_query_returns_self_with_metadata(kind, params):
    c, X, ids, metas = build(kind, **params)
    extra = {"nprobe": 8} if kind == "ivf" else {}
    hits = c.query(X[:5], k=3, **extra)
    for i, row in enumerate(hits):
        assert row[0].id == ids[i] and row[0].distance == pytest.approx(0, abs=1e-5)
        assert row[0].metadata == metas[i]


@pytest.mark.parametrize("kind,params", [("flat", {}), ("hnsw", {"M": 8, "ef_construction": 60})])
def test_filtered_query_obeys_filter(kind, params):
    c, X, ids, metas = build(kind, **params)
    where = {"topic": {"$in": [1, 2]}, "year": {"$gte": 2010}}
    extra = {"ef": 100} if kind == "hnsw" else {}
    for row in c.query(X[:10], k=5, where=where, **extra):
        assert row and all(h.metadata["topic"] in (1, 2) and h.metadata["year"] >= 2010 for h in row)
    assert c.count(where) == sum(1 for m in metas if m["topic"] in (1, 2) and m["year"] >= 2010)
    assert c.query(X[:1], k=3, where={"topic": 999}) == [[]]


def test_upsert_replaces_and_delete_removes():
    c = Collection(2, "l2", "flat")
    c.upsert(["a", "b"], [[0, 0], [10, 10]], [{"v": 1}, {"v": 1}])
    c.upsert("a", [[9, 9]], [{"v": 2}])                   # move 'a' next to 'b'
    assert len(c) == 2 and c.stats()["dead_rows"] == 1
    vec, meta = c.get("a")
    assert vec.tolist() == [9.0, 9.0] and meta == {"v": 2}
    assert c.query([[0, 0]], k=1)[0][0].id == "a"          # the moved vector, not the stale one
    assert [h.id for h in c.query([[9.2, 9.2]], k=2)[0]] == ["a", "b"]
    assert c.delete(["a", "zzz"]) == 1
    assert "a" not in c and [h.id for h in c.query([[9, 9]], k=5)[0]] == ["b"]
    with pytest.raises(KeyError):
        c.get("a")


def test_metadata_filter_cache_invalidates_on_write():
    c = Collection(2, "l2", "flat")
    c.upsert(["a"], [[0, 0]], [{"k": 1}])
    assert c.count({"k": 1}) == 1
    c.upsert(["b"], [[1, 1]], [{"k": 1}])
    assert c.count({"k": 1}) == 2
    c.delete("a")
    assert c.count({"k": 1}) == 1


def test_validation_is_atomic():
    c = Collection(3, "cosine", "flat")
    c.upsert(["a"], [[1, 0, 0]])
    with pytest.raises(ValueError):
        c.upsert(["b", "c"], [[1, 1, 1], [0, 0, 0]])      # zero vector under cosine
    assert len(c) == 1 and c.index.ntotal == 1
    with pytest.raises(ValueError):
        c.upsert(["x", "x"], [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        c.upsert(["y"], [[1, 0, 0]], [{"bad": object()}])
    with pytest.raises(ValueError):
        c.upsert([""], [[1, 0, 0]])
    assert len(c) == 1


def test_compact_reclaims_dead_rows():
    c, X, ids, _ = build("flat", n=100)
    c.delete(ids[:60])
    assert c.stats()["dead_rows"] == 60
    c.compact()
    assert c.stats()["dead_rows"] == 0 and len(c) == 40
    assert c.query(X[70:71], k=1)[0][0].id == ids[70]


@pytest.mark.parametrize("kind,params", [("flat", {}), ("hnsw", {"M": 8, "ef_construction": 60}), ("ivf", {"nlist": 8})])
def test_collection_save_load_roundtrip(tmp_path, kind, params):
    c, X, ids, metas = build(kind, **params)
    c.delete(ids[:5])
    extra = {"nprobe": 8} if kind == "ivf" else {}
    before = c.query(X[:20], k=5, where={"topic": {"$lt": 4}}, **extra)
    c.save(str(tmp_path))
    d = Collection.load(str(tmp_path))
    after = d.query(X[:20], k=5, where={"topic": {"$lt": 4}}, **extra)
    assert [[(h.id, round(h.distance, 5)) for h in r] for r in before] == [[(h.id, round(h.distance, 5)) for h in r] for r in after]
    assert len(d) == len(c) and d.get(ids[10])[1] == metas[10]
    assert ids[0] not in d
    d.upsert("new", X[:1] + 0.01, [{"topic": 0, "year": 2001}])   # still writable after load
    assert "new" in d


def test_index_roundtrip_each_kind(tmp_path):
    X, _ = clustered(300, 8, n_clusters=5, seed=0)
    Q = X[:10] + 0.01
    for idx in (FlatIndex(8), HNSWIndex(8, M=8, ef_construction=50), IVFIndex(8, nlist=6),
                IVFIndex(8, nlist=6, pq_m=4, pq_bits=5, keep_raw=True)):
        idx.add(X)
        if isinstance(idx, IVFIndex):
            idx.train()
        p = str(tmp_path / f"{idx.kind}.npz")
        save_index(idx, p)
        back = load_index(p)
        kw = {"nprobe": 6} if isinstance(idx, IVFIndex) else {}
        a, b = idx.search(Q, 5, **kw), back.search(Q, 5, **kw)
        assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


def test_untrained_ivf_roundtrip_keeps_pending(tmp_path):
    X, _ = clustered(100, 8, n_clusters=3, seed=0)
    idx = IVFIndex(8, nlist=4)
    idx.add(X)
    save_index(idx, str(tmp_path / "i.npz"))
    back = load_index(str(tmp_path / "i.npz"))
    assert not back.trained and back.ntotal == 100
    back.train()
    assert back.list_sizes().sum() == 100


def test_hnsw_resumes_building_after_load(tmp_path):
    X, _ = clustered(300, 8, n_clusters=5, seed=0)
    h = HNSWIndex(8, M=8, ef_construction=50)
    h.add(X[:200])
    save_index(h, str(tmp_path / "h.npz"))
    back = load_index(str(tmp_path / "h.npz"))
    back.add(X[200:])
    assert back.ntotal == 300 and back.search(X[250:251], 1)[0][0, 0] == 250


def test_load_rejects_unknown_format(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, __meta__=np.frombuffer(b'{"kind":"flat","format":99}', dtype=np.uint8))
    with pytest.raises(ValueError):
        load_index(str(p))
