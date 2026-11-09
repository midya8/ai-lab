import numpy as np
from datadoctor.synth import make_blobs, inject_instance_noise


def test_exact_rate_and_labels_change():
    X, y = make_blobs(n=600, seed=1)
    yn, flip = inject_instance_noise(X, y, 0.2, seed=1)
    assert flip.sum() == 120
    assert (yn[flip] != y[flip]).all() and (yn[~flip] == y[~flip]).all()


def test_deterministic():
    X, y = make_blobs(n=400, seed=2)
    a = inject_instance_noise(X, y, 0.15, seed=3)[0]
    b = inject_instance_noise(X, y, 0.15, seed=3)[0]
    assert (a == b).all()


def test_errors_concentrate_near_borders():
    X, y = make_blobs(n=1500, seed=0)
    _, flip = inject_instance_noise(X, y, 0.1, seed=0)
    cent = np.stack([X[y == c].mean(0) for c in range(5)])
    d = np.linalg.norm(X[:, None] - cent[None], axis=2)
    own = d[np.arange(len(y)), y]
    d2 = d.copy(); d2[np.arange(len(y)), y] = np.inf
    gap = own - d2.min(1)
    assert gap[flip].mean() > gap[~flip].mean() + 0.5


def test_flip_goes_to_nearest_other_class():
    X, y = make_blobs(n=500, seed=4)
    yn, flip = inject_instance_noise(X, y, 0.1, seed=4)
    cent = np.stack([X[y == c].mean(0) for c in range(5)])
    for i in np.flatnonzero(flip):
        d = np.linalg.norm(cent - X[i], axis=1); d[y[i]] = np.inf
        assert yn[i] == d.argmin()
