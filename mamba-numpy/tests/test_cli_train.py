import numpy as np
import pytest

from mamba import Mamba, selective_copy
from mamba.__main__ import main
from mamba.train import evaluate, fit


def test_checkpoint_roundtrip(tmp_path):
    m = Mamba(vocab=7, d_model=6, d_state=3, n_layer=2, seed=4)
    toks = np.random.default_rng(0).integers(0, 7, (2, 9))
    m.save(tmp_path / "m.npz")
    m2 = Mamba.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(m.forward(toks), m2.forward(toks))
    assert m2.cfg == m.cfg


def test_checkpoint_roundtrip_lti_and_shape_check(tmp_path):
    m = Mamba(vocab=7, d_model=6, d_state=3, selective=False, seed=4)
    m.save(tmp_path / "m.npz")
    assert Mamba.load(tmp_path / "m.npz").cfg["selective"] is False
    d = dict(np.load(tmp_path / "m.npz"))
    d["emb"] = np.zeros((3, 3))
    np.savez(tmp_path / "bad.npz", **d)
    with pytest.raises(ValueError):
        Mamba.load(tmp_path / "bad.npz")


def test_training_reduces_loss_and_is_deterministic():
    def run():
        m = Mamba(vocab=6, d_model=8, d_state=4, n_layer=1, seed=0)
        mk = lambda n, rng: selective_copy(n, 8, 2, 4, rng)
        h = fit(m, mk, steps=60, batch=16, lr=5e-3, log_every=59)
        return h, m
    (h1, m1), (h2, m2) = run(), run()
    assert h1[-1]["loss"] < h1[0]["loss"]
    assert h1[-1]["loss"] == h2[-1]["loss"]
    np.testing.assert_array_equal(m1.params()["emb"], m2.params()["emb"])


def test_selective_model_learns_a_small_copy_task():
    m = Mamba(vocab=6, d_model=12, d_state=6, n_layer=2, seed=0)
    mk = lambda n, rng: selective_copy(n, 8, 2, 4, rng)
    fit(m, mk, steps=500, batch=32, lr=4e-3)
    tok, _ = evaluate(m, mk(500, np.random.default_rng(5)))
    assert tok > 0.6  # chance is 0.25


def test_cli_train_then_eval(tmp_path, capsys):
    ck = str(tmp_path / "c.npz")
    assert main(["train", "--steps", "20", "--seq-len", "8", "--n-data", "2", "--d-model", "8", "--save", ck]) == 0
    assert main(["eval", ck, "--seq-len", "8", "--n-data", "2", "--n", "50"]) == 0
    out = capsys.readouterr().out
    assert "token accuracy" in out
    assert main(["eval", ck, "--task", "recall", "--n-data", "2"]) == 2
