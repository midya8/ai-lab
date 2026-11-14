import numpy as np
from datadoctor import drift_report


def _corr(rng, rho, n=600):
    return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)


def test_none_on_same_distribution():
    rng = np.random.default_rng(1)
    r = drift_report(rng.normal(size=(500, 3)), rng.normal(size=(500, 3)))
    assert r["verdict"] == "none" and r["flagged_features"] == []


def test_marginal_shift():
    rng = np.random.default_rng(2)
    b = rng.normal(size=(500, 3)); b[:, 1] += 1.5
    r = drift_report(rng.normal(size=(500, 3)), b, names=["a", "b", "c"])
    assert r["verdict"] in ("marginal", "both") and "b" in r["flagged_features"]


def test_joint_only_shift():
    rng = np.random.default_rng(3)
    r = drift_report(_corr(rng, 0.0), _corr(rng, 0.9), n_perm=200)
    assert r["verdict"] == "joint-only" and r["mmd"]["drifted"]
