import numpy as np

from visionlab.data import CLASSES, ascii_art, augment, make_dataset, render


def test_dataset_is_deterministic_balanced_and_in_range():
    X1, y1 = make_dataset(60, seed=3)
    X2, y2 = make_dataset(60, seed=3)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert X1.shape == (60, 1, 24, 24) and X1.dtype == np.float32
    assert X1.min() >= 0 and X1.max() <= 1
    assert np.bincount(y1).tolist() == [10] * len(CLASSES)
    X3, _ = make_dataset(60, seed=4)
    assert not np.array_equal(X1, X3)


def test_every_class_draws_a_visible_shape():
    rng = np.random.default_rng(0)
    for c in CLASSES:
        img = render(c, rng, noise=0.0)
        assert (img > 0.5).sum() > 20, c


def test_shapes_have_expected_relative_areas():
    """Noise-free area sanity: ring < disk, cross < square, triangle < disk (same radius scale)."""
    area = {}
    for c in CLASSES:
        rng = np.random.default_rng(7)  # same radius / pose for each class
        area[c] = render(c, rng, noise=0.0).sum()
    assert area["ring"] < area["disk"]
    assert area["triangle"] < area["disk"]
    assert area["stripes"] < area["square"]


def test_augment_preserves_pixel_multiset_and_shape():
    X, _ = make_dataset(20, seed=1)
    A = augment(X, np.random.default_rng(0))
    assert A.shape == X.shape
    for a, x in zip(A, X):
        np.testing.assert_array_equal(np.sort(a.ravel()), np.sort(x.ravel()))


def test_ascii_art_dimensions():
    art = ascii_art(np.zeros((1, 6, 4)))
    assert len(art.splitlines()) == 6 and len(art.splitlines()[0]) == 8
