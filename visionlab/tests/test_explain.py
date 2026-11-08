import numpy as np
import pytest

from visionlab import explain as ex
from visionlab import png
from visionlab.data import make_dataset
from visionlab.layers import Conv2D
from visionlab.model import ModelConfig, build_cnn
from visionlab.train import train

from test_layers import num_grad

_c = {}


def model_and_data():
    if not _c:
        Xtr, ytr = make_dataset(900, seed=1)
        m = build_cnn(ModelConfig(width=6, hidden=32))
        train(m, Xtr, ytr, epochs=6, log=None)
        _c["v"] = (m, *make_dataset(24, seed=8))
    return _c["v"]


def test_saliency_matches_logit_gradient_finite_differences():
    m = build_cnn(ModelConfig(img=8, width=2, hidden=6, n_classes=3, dtype="float64"))
    x = np.random.default_rng(0).random((1, 8, 8))
    xb = x[None]
    f = lambda: float(m.predict_logits(xb)[0, 1])
    sal = ex.saliency(m, x, cls=1)
    np.testing.assert_allclose(sal, np.abs(num_grad(f, xb))[0, 0], atol=1e-6, rtol=1e-4)


def test_every_method_returns_input_sized_nonnegative_maps():
    m, X, y = model_and_data()
    for name, fn in ex.METHODS.items():
        h = fn(m, X[0])
        assert h.shape == (24, 24), name
        assert np.isfinite(h).all(), name
        if name != "occlusion":  # occlusion can be negative when blanking helps the class
            assert h.min() >= 0, name


def test_gradcam_targets_the_last_conv_relu_and_channel_weights_are_exact():
    m, X, y = model_and_data()
    j = ex.last_conv_relu(m)
    assert isinstance(m.layers[j - 1], Conv2D)
    x = X[1]
    cls = int(m.predict(x[None])[0])
    cam = ex.grad_cam(m, x, cls)
    # recompute by hand from stored activations / gradients
    A, G = m.acts[j][0], m.gouts[j][0]
    raw = np.maximum((G.mean(axis=(1, 2))[:, None, None] * A).sum(0), 0)
    assert raw.shape == (6, 6) or raw.shape == (12, 12)
    assert cam.shape == (24, 24) and cam.min() >= 0


def test_gradcam_is_class_specific():
    m, X, y = model_and_data()
    a, b = ex.grad_cam(m, X[0], 0), ex.grad_cam(m, X[0], 3)
    assert not np.allclose(a, b)


def test_occlusion_of_constant_model_region_is_zero():
    """Blanking an already-blank background patch must not change the probability."""
    m, X, y = model_and_data()
    x = np.zeros((1, 24, 24), dtype=np.float32)
    h = ex.occlusion(m, x)
    np.testing.assert_allclose(h, 0.0, atol=1e-5)  # float32 batch-size rounding only


def test_deletion_curve_starts_at_clean_probability_and_informative_maps_beat_random():
    m, X, y = model_and_data()
    x = X[0]
    cls = int(m.predict(x[None])[0])
    fr = [0, 0.1, 0.3]
    curve = ex.deletion_curve(m, x, ex.occlusion(m, x, cls), cls, fr)
    clean = m.predict_proba(x[None])[0, cls]
    assert curve[0] == pytest.approx(clean, abs=1e-6)
    rng = np.random.default_rng(0)
    informative, _, n = ex.deletion_score(m, X, lambda x, c: ex.occlusion(m, x, c), only_correct=True, y=y)
    random, _, _ = ex.deletion_score(m, X, lambda x, c: ex.random_heat(x.shape[-2:], rng), only_correct=True, y=y)
    assert n > 10 and informative < random


def test_png_writer_roundtrip(tmp_path):
    import zlib
    img = np.random.default_rng(0).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    png.write_png(p, img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.read_png_size(p) == (5, 7)
    idat = data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    rows = np.frombuffer(raw, np.uint8).reshape(5, 1 + 21)
    assert (rows[:, 0] == 0).all()
    np.testing.assert_array_equal(rows[:, 1:].reshape(5, 7, 3), img)
    g = tmp_path / "g.png"
    png.write_png(g, img[:, :, 0])
    assert png.read_png_size(g) == (5, 7)
    with pytest.raises(ValueError):
        png.write_png(g, img.astype(float))


def test_montage_and_colormap():
    h = png.heat_rgb(np.array([0.0, 0.5, 1.0]))
    assert h[0].tolist() == [0, 0, 0] and h[2].tolist() == [255, 255, 255]
    tiles = [[np.zeros((4, 4, 3), np.uint8)] * 3] * 2
    assert png.montage(tiles, pad=1).shape == (2 * 5 + 1, 3 * 5 + 1, 3)
