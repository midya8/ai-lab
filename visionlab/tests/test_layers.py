import numpy as np
import pytest

from visionlab.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, Normalize, ReLU


def num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        hi = f()
        x[i] = old - eps
        lo = f()
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def check_layer(layer, x, rng):
    """Finite-difference check of dx and every parameter gradient for loss = sum(w * out)."""
    out = layer.forward(x)
    w = rng.standard_normal(out.shape)
    dx = layer.backward(w)
    loss = lambda: float((layer.forward(x) * w).sum())
    np.testing.assert_allclose(dx, num_grad(loss, x), atol=1e-6, rtol=1e-5)
    layer.forward(x)
    layer.backward(w)
    for name, p in layer.params().items():
        np.testing.assert_allclose(layer.grads()[name], num_grad(loss, p), atol=1e-6, rtol=1e-5, err_msg=name)


def naive_conv(x, W, b, pad):
    n, c, h, w = x.shape
    f, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, f, h + 2 * pad - k + 1, w + 2 * pad - k + 1))
    for i in range(out.shape[2]):
        for j in range(out.shape[3]):
            patch = xp[:, :, i:i + k, j:j + k]
            out[:, :, i, j] = np.tensordot(patch, W, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


@pytest.mark.parametrize("pad", [0, 1, 2])
def test_conv_matches_naive_loops(pad):
    rng = np.random.default_rng(0)
    conv = Conv2D(3, 4, 3, pad, rng, np.float64)
    conv.b[...] = rng.standard_normal(4)
    x = rng.standard_normal((2, 3, 7, 6))
    np.testing.assert_allclose(conv.forward(x), naive_conv(x, conv.W, conv.b, pad), atol=1e-12)


@pytest.mark.parametrize("pad,k", [(0, 3), (1, 3), (2, 5)])
def test_conv_gradients(pad, k):
    rng = np.random.default_rng(1)
    conv = Conv2D(2, 3, k, pad, rng, np.float64)
    check_layer(conv, rng.standard_normal((2, 2, 6, 7)), rng)


def test_dense_gradients():
    rng = np.random.default_rng(2)
    check_layer(Dense(5, 4, rng, np.float64), rng.standard_normal((3, 5)), rng)


def test_relu_and_flatten_gradients():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4)) + 0.05  # keep away from the kink
    x[np.abs(x) < 1e-3] = 0.5
    check_layer(ReLU(), x, rng)
    check_layer(Flatten(), rng.standard_normal((2, 3, 4, 4)), rng)


def test_maxpool_forward_and_gradient_even_and_odd():
    rng = np.random.default_rng(4)
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(MaxPool2D().forward(x)[0, 0], [[5, 7], [13, 15]])
    for shape in [(2, 2, 6, 6), (1, 2, 5, 7)]:  # odd sizes crop the last row/col
        pool = MaxPool2D()
        x = rng.standard_normal(shape)
        out = pool.forward(x)
        assert out.shape == (shape[0], shape[1], shape[2] // 2, shape[3] // 2)
        check_layer(pool, x, rng)


def test_maxpool_ties_send_gradient_to_exactly_one_pixel():
    pool = MaxPool2D()
    x = np.ones((1, 1, 2, 2))
    pool.forward(x)
    dx = pool.backward(np.ones((1, 1, 1, 1)))
    assert dx.sum() == 1.0 and (dx > 0).sum() == 1


def test_normalize_gradient():
    rng = np.random.default_rng(5)
    check_layer(Normalize(0.3, 0.7), rng.standard_normal((2, 1, 3, 3)), rng)


def test_dropout_train_vs_eval_and_scaling():
    d = Dropout(0.5, np.random.default_rng(0))
    x = np.ones((200, 50))
    y = d.forward(x)
    assert set(np.unique(y)) <= {0.0, 2.0}
    assert abs(y.mean() - 1.0) < 0.05          # inverted dropout preserves the mean
    np.testing.assert_array_equal(d.backward(np.ones_like(x)), y)  # same mask on the way back
    d.training = False
    np.testing.assert_array_equal(d.forward(x), x)
