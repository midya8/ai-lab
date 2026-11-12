import numpy as np
from datadoctor import suggest_relabels


def _data():
    P = np.array([[0.05, 0.95], [0.6, 0.4], [0.9, 0.1], [0.45, 0.55]])
    y = np.array([0, 0, 0, 0])
    return P, y


def test_only_confident_flags_suggested():
    P, y = _data()
    mask = np.array([True, True, False, True])
    r = suggest_relabels(P, y, mask)
    assert [x["index"] for x in r] == [0]
    assert r[0]["suggested"] == 1 and abs(r[0]["margin"] - 0.9) < 1e-9


def test_thresholds_loosen():
    P, y = _data()
    mask = np.array([True, False, False, True])
    r = suggest_relabels(P, y, mask, min_confidence=0.5, min_margin=0.05)
    assert [x["index"] for x in r] == [0, 3]


def test_unflagged_never_suggested():
    P, y = _data()
    assert suggest_relabels(P, y, np.zeros(4, bool)) == []
