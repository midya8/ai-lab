import numpy as np
import pytest

from visionlab.model import ModelConfig, Sequential, build_cnn, cross_entropy, softmax

from test_layers import num_grad


def tiny(**kw):
    return build_cnn(ModelConfig(img=8, width=2, hidden=6, n_classes=3, dtype="float64", **kw))


def test_softmax_rows_sum_to_one_and_stable():
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-12


def test_cross_entropy_gradient_with_label_smoothing():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 3))
    y = np.array([0, 2, 1, 1])
    for ls in (0.0, 0.1):
        _, g = cross_entropy(z, y, ls)
        np.testing.assert_allclose(g, num_grad(lambda: cross_entropy(z, y, ls)[0], z), atol=1e-7)


def test_whole_network_gradcheck_every_parameter():
    rng = np.random.default_rng(1)
    m = tiny()
    m.eval()
    x = rng.random((3, 1, 8, 8))
    y = np.array([0, 1, 2])
    loss, dl = cross_entropy(m.forward(x), y)
    dx = m.backward(dl)
    f = lambda: cross_entropy(m.forward(x), y)[0]
    np.testing.assert_allclose(dx, num_grad(f, x), atol=1e-6, rtol=1e-4)
    m.forward(x)
    m.backward(cross_entropy(m.forward(x), y)[1])
    for name, p, g in m.parameters():
        np.testing.assert_allclose(g, num_grad(f, p), atol=1e-6, rtol=1e-4, err_msg=name)


def test_dropout_network_backward_uses_the_forward_mask():
    m = tiny(dropout=0.5)
    m.train()
    x = np.random.default_rng(2).random((2, 1, 8, 8))
    m.forward(x)
    m.backward(np.ones((2, 3)))  # must not raise; mask cached from the forward pass
    m.eval()
    a, b = m.forward(x), m.forward(x)
    np.testing.assert_array_equal(a, b)  # deterministic in eval mode


def test_save_load_roundtrip(tmp_path):
    m = build_cnn(ModelConfig(width=4, hidden=16, seed=3))
    x = np.random.default_rng(0).random((5, 1, 24, 24)).astype(np.float32)
    path = tmp_path / "m.npz"
    m.save(path)
    m2 = Sequential.load(path)
    np.testing.assert_array_equal(m.predict_logits(x), m2.predict_logits(x))
    assert m2.config == m.config


def test_img_must_be_divisible_by_four():
    with pytest.raises(ValueError):
        build_cnn(ModelConfig(img=10))


def test_predict_restores_training_flags():
    m = tiny(dropout=0.3)
    m.train()
    m.predict(np.zeros((1, 1, 8, 8)))
    assert all(l.training for l in m.layers)
