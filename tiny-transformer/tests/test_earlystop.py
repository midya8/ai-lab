import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt.earlystop import EarlyStopper  # noqa: E402
from tinygpt import train  # noqa: E402


def test_stops_after_patience():
    es, p = EarlyStopper(patience=2), {"w": np.zeros(1)}
    assert not es.update(1.0, p, 1)
    assert not es.update(1.1, p, 2)
    assert es.update(1.2, p, 3)
    assert es.best_step == 1


def test_improvement_resets_counter():
    es, p = EarlyStopper(patience=2), {"w": np.zeros(1)}
    es.update(1.0, p, 1)
    es.update(1.5, p, 2)
    assert not es.update(0.9, p, 3)
    assert es.bad == 0 and es.best_step == 3


def test_min_delta():
    es, p = EarlyStopper(patience=1, min_delta=0.1), {"w": np.zeros(1)}
    es.update(1.0, p, 1)
    assert es.update(0.95, p, 2)


def test_restore_best_weights_in_place():
    es, p = EarlyStopper(patience=3), {"w": np.array([1.0])}
    es.update(1.0, p, 1)
    p["w"][...] = 99.0
    es.update(2.0, p, 2)
    ref = p["w"]
    assert es.restore(p) and ref[0] == 1.0


def test_restore_empty_and_bad_patience():
    assert not EarlyStopper().restore({"w": np.zeros(1)})
    with pytest.raises(ValueError):
        EarlyStopper(patience=0)


def test_train_cli_early_stops(tmp_path, capsys):
    data = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    train.main(["--data", data, "--steps", "400", "--eval-every", "10", "--patience", "1",
                "--lr", "0.5", "--embd", "16", "--layers", "1", "--heads", "2",
                "--out", str(tmp_path / "m.npz"), "--sample", "5"])
    out = capsys.readouterr().out
    assert "early stop" in out and "held-out loss" in out
