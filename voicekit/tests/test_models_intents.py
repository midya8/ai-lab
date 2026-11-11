import numpy as np
import pytest

from voicekit.intents import group_events, parse_events, parse_words
from voicekit.models import DTWClassifier, MLP, dtw_distance
from voicekit.pipeline import Event


def blobs(n=200, k=3, d=6, seed=0, spread=0.6):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, d)) * 3
    y = rng.integers(0, k, n)
    return centres[y] + spread * rng.standard_normal((n, d)), y


def test_mlp_learns_blobs_and_loss_decreases():
    X, y = blobs()
    m = MLP(hidden=16, epochs=40, seed=1).fit(X, y)
    assert m.loss_history[-1] < m.loss_history[0] * 0.3
    Xt, yt = blobs(seed=0)  # same centres
    assert (m.predict(Xt) == yt).mean() > 0.95


def test_mlp_learns_xor():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    m = MLP(hidden=32, epochs=150, lr=1e-2, dropout=0.0, seed=0).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.93


def test_mlp_probabilities_normalised_and_save_load(tmp_path):
    X, y = blobs()
    m = MLP(hidden=8, epochs=5).fit(X, y)
    p = m.predict_proba(X[:10])
    assert np.allclose(p.sum(1), 1) and (p >= 0).all()
    m.save(str(tmp_path / "m.npz"))
    m2 = MLP.load(str(tmp_path / "m.npz"))
    assert np.allclose(m2.predict_proba(X[:10]), p)


def test_mlp_unfitted_raises_and_gradient_matches_finite_difference():
    with pytest.raises(RuntimeError):
        MLP().predict_proba(np.zeros((1, 3)))
    # finite-difference check of the manual backprop on one batch (dropout off)
    X, y = blobs(n=12, d=4, seed=3)
    m = MLP(hidden=5, epochs=0, dropout=0.0, weight_decay=0.0)
    m.fit(X, y, n_classes=3)  # initialises params
    Xn = (X - m.mu) / m.sd

    def _sm(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def loss():
        return float(-np.log(_sm(m._forward(Xn)[3])[np.arange(12), y]).mean())

    z1, h, _, logits = m._forward(Xn)
    g = _sm(logits); g[np.arange(12), y] -= 1; g /= 12
    analytic = Xn.T @ ((g @ m.params["W2"].T) * (z1 > 0))
    num = np.zeros_like(analytic)
    for i in range(analytic.shape[0]):
        for j in range(analytic.shape[1]):
            old = m.params["W1"][i, j]
            m.params["W1"][i, j] = old + 1e-5; lp = loss()
            m.params["W1"][i, j] = old - 1e-5; lm = loss()
            m.params["W1"][i, j] = old
            num[i, j] = (lp - lm) / 2e-5
    assert np.allclose(analytic, num, atol=1e-6)


def test_dtw_identity_symmetry_and_warp_tolerance():
    t = np.linspace(0, 1, 40)
    a = np.stack([np.sin(6 * t), np.cos(4 * t)], 1)
    b = np.stack([np.sin(6 * t**1.3), np.cos(4 * t**1.3)], 1)  # time-warped copy
    c = np.stack([np.sin(6 * t + 2), -np.cos(4 * t)], 1)       # different shape
    assert dtw_distance(a, a) == pytest.approx(0)
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))
    assert dtw_distance(a, b) < dtw_distance(a, c)
    # the warped copy is closer under DTW than under rigid frame-by-frame matching
    rigid = np.sqrt(((a - b) ** 2).sum(1)).sum() / (2 * len(a))
    assert dtw_distance(a, b) < rigid


def test_dtw_handles_unequal_lengths():
    a, b = np.random.default_rng(0).standard_normal((30, 3)), np.random.default_rng(1).standard_normal((12, 3))
    assert np.isfinite(dtw_distance(a, b))


def test_dtw_classifier_separates_shapes():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 30)

    def make(c):
        n = rng.integers(25, 35)
        tt = np.linspace(0, 1, n)
        return np.stack([np.sin((c + 1) * 4 * tt), tt * (c - 1)], 1) + 0.05 * rng.standard_normal((n, 2))

    y = np.array([i % 3 for i in range(60)])
    seqs = [make(c) for c in y]
    clf = DTWClassifier(per_class=3).fit(seqs, y)
    assert len(clf.templates) == 9
    assert np.mean([clf.predict(make(c), 3) == c for c in [0, 1, 2] * 5]) == 1.0


# --- intents ------------------------------------------------------------------------------

@pytest.mark.parametrize("words,name", [
    (["lights", "on"], "lights.on"), (["off", "lights"], "lights.off"), (["timer", "on"], "timer.on"),
    (["off", "timer"], "timer.off"), (["play", "music"], "music.play"), (["music", "stop"], "music.stop"),
    (["stop", "music"], "music.stop"), (["music", "play"], "music.play"), (["weather"], "weather.query"),
    (["play"], "music.play"), (["stop"], "stop_all"), (["lights", "lights", "on"], "lights.on"),
])
def test_valid_commands(words, name):
    assert parse_words(words).name == name


@pytest.mark.parametrize("words", [[], ["lights"], ["on"], ["lights", "play"], ["lights", "timer", "on"],
                                   ["weather", "on"], ["on", "off"], ["music", "on"], ["bogus"]])
def test_invalid_commands(words):
    assert parse_words(words) is None


def ev(word, a, b):
    return Event(word, 1.0, a, b)


def test_grouping_by_gap():
    events = [ev("lights", 1.0, 1.4), ev("on", 1.7, 1.9), ev("stop", 5.0, 5.3), ev("weather", 5.5, 6.0)]
    groups = group_events(events, window_s=1.0)
    assert [[e.word for e in g] for g in groups] == [["lights", "on"], ["stop", "weather"]]
    parsed = parse_events(events, window_s=1.0)
    assert parsed[0][1].name == "lights.on" and parsed[1][1] is None


def test_grouping_sorts_unordered_events():
    groups = group_events([ev("on", 1.7, 1.9), ev("lights", 1.0, 1.4)])
    assert [e.word for e in groups[0]] == ["lights", "on"]
