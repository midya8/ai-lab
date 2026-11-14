import numpy as np
import pytest
from datadoctor.psiboot import psi_ci, psi_ci_report


def test_ci_brackets_and_orders():
    r = np.random.default_rng(0)
    out = psi_ci(r.normal(size=500), r.normal(0.5, 1, size=500))
    assert out["lo"] <= out["hi"]
    assert out["lo"] > 0


def test_small_sample_ci_is_wider():
    r = np.random.default_rng(1)
    big = psi_ci(r.normal(size=2000), r.normal(size=2000))
    small = psi_ci(r.normal(size=40), r.normal(size=40))
    assert small["hi"] - small["lo"] > big["hi"] - big["lo"]


def test_report_flags_only_drifted_feature():
    r = np.random.default_rng(2)
    A = r.normal(size=(800, 2))
    B = r.normal(size=(800, 2)); B[:, 1] += 1.5
    rep = psi_ci_report(A, B, ["same", "moved"])
    assert rep[0]["feature"] == "moved" and rep[0]["confident"]
    assert not rep[1]["confident"]


def test_validation():
    with pytest.raises(ValueError):
        psi_ci([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        psi_ci([1, 2, 3], [1, 2, 3], level=1.5)
