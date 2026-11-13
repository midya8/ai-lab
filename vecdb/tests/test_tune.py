import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest

from vecdb import Collection, gaussian


def build(kind, n=1500, **kw):
    X = gaussian(n, 16, seed=3)
    c = Collection(16, "cosine", kind, **kw)
    if kind == "ivf":
        c.train(X)
    c.upsert([f"v{i}" for i in range(n)], X)
    return c


def test_hnsw_tune_reaches_target_and_curve_monotone_stop():
    c = build("hnsw", M=12, ef_construction=80)
    r = c.tune(k=5, target_recall=0.9)
    assert r["param"] == "ef" and r["reached"] and r["recall"] >= 0.9
    assert r["curve"][-1] == (r["value"], r["recall"])
    assert all(a[0] < b[0] for a, b in zip(r["curve"], r["curve"][1:]))


def test_ivf_tune_full_probe_is_exact():
    c = build("ivf", nlist=8)
    r = c.tune(k=5, target_recall=0.999, grid=[1, 2, 8])
    assert r["param"] == "nprobe" and r["reached"] and r["value"] <= 8


def test_unreachable_target_reports_best():
    c = build("hnsw", M=12, ef_construction=80)
    r = c.tune(k=5, target_recall=1.01, grid=[5, 10])
    assert not r["reached"] and r["value"] == 10


def test_flat_and_empty():
    assert build("flat", n=50).tune()["recall"] == 1.0
    with pytest.raises(ValueError):
        Collection(4, "cosine", "hnsw").tune()
