import numpy as np
import pytest

from vecdb import HybridCollection


def build():
    rng = np.random.default_rng(1)
    h = HybridCollection(8)
    V = rng.normal(size=(4, 8)).astype(np.float32)
    h.upsert(["a", "b", "c", "d"], V, [{"g": 0}, {"g": 1}, {"g": 0}, {"g": 1}],
             ["error code E42 auth", "fox jumps", "auth module notes", "lazy dog"])
    return h, V


def test_roundtrip_same_results(tmp_path):
    h, V = build()
    h.delete("d")
    h.save(str(tmp_path))
    h2 = HybridCollection.load(str(tmp_path))
    assert len(h2) == 3
    for fusion in ("rrf", "weighted"):
        a = h.query(V[0], "auth E42", k=3, fusion=fusion)
        b = h2.query(V[0], "auth E42", k=3, fusion=fusion)
        assert [x.id for x in a] == [x.id for x in b]
        assert [x.distance for x in a] == pytest.approx([x.distance for x in b])
    assert h2.lexical("lazy dog") == []  # deleted text did not come back


def test_filters_survive_load(tmp_path):
    h, V = build()
    h.save(str(tmp_path))
    h2 = HybridCollection.load(str(tmp_path))
    hits = h2.query(V[1], "fox", k=4, where={"g": 0})
    assert hits and all(x.metadata["g"] == 0 for x in hits)


def test_upsert_after_load(tmp_path):
    h, V = build()
    h.save(str(tmp_path))
    h2 = HybridCollection.load(str(tmp_path))
    h2.upsert(["e"], V[:1], [{"g": 0}], ["zebra crossing"])
    assert h2.lexical("zebra")[0][0] == "e"


def test_corrupt_texts_rejected(tmp_path):
    h, _ = build()
    h.save(str(tmp_path))
    (tmp_path / "texts.json").write_text('{"ghost": "x"}')
    with pytest.raises(ValueError):
        HybridCollection.load(str(tmp_path))
