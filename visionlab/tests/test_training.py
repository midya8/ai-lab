import numpy as np

from visionlab.attacks import fgsm, loss_grad, pgd
from visionlab.data import make_dataset
from visionlab.metrics import confusion_matrix, per_class, wilson
from visionlab.model import ModelConfig, build_cnn, cross_entropy
from visionlab.optim import SGD, Adam
from visionlab.robust import attack_report, natural_report, shift
from visionlab.train import train

from test_layers import num_grad


def trained_model():
    Xtr, ytr = make_dataset(1500, seed=1)
    Xva, yva = make_dataset(240, seed=2)
    m = build_cnn(ModelConfig(width=6, hidden=32))
    hist = train(m, Xtr, ytr, Xva, yva, epochs=8, log=None)
    return m, hist, Xva, yva


_cache = {}


def get_trained():
    if "m" not in _cache:
        _cache["m"] = trained_model()
    return _cache["m"]


def test_training_learns_the_task_and_loss_falls():
    m, hist, Xva, yva = get_trained()
    assert hist[-1]["loss"] < hist[0]["loss"] * 0.6
    assert hist[-1]["val_acc"] > 0.7       # chance is 1/6
    assert all(not l.training for l in m.layers)


class _Quad:
    """Stand-in model whose loss is |w - target|^2, to test optimisers in isolation."""

    def __init__(self):
        self.target = np.array([[3.0, -2.0], [0.5, 1.0]])
        self.w = np.zeros((2, 2))
        self.g = np.zeros((2, 2))

    def parameters(self):
        return [("0.Quad.W", self.w, self.g)]

    def loss(self):
        self.g[...] = 2 * (self.w - self.target)
        return float(((self.w - self.target) ** 2).sum())


def test_optimisers_converge_on_a_quadratic():
    for make in (lambda m: SGD(m, lr=0.05, momentum=0.9), lambda m: Adam(m, lr=0.1)):
        q = _Quad()
        opt = make(q)
        first = q.loss()
        for _ in range(300):
            q.loss()
            opt.step()
        assert q.loss() < 1e-3 * first


def test_weight_decay_shrinks_weights_only():
    q = _Quad()
    q.w[...] = 1.0
    q.g[...] = 0.0
    Adam(q, lr=0.1, weight_decay=0.5).step()
    assert (q.w < 1.0).all()  # decoupled decay applied to '.W' params


def test_loss_grad_matches_finite_differences():
    m = build_cnn(ModelConfig(img=8, width=2, hidden=6, n_classes=3, dtype="float64"))
    x = np.random.default_rng(0).random((2, 1, 8, 8))
    y = np.array([0, 2])
    g = loss_grad(m, x, y)
    np.testing.assert_allclose(g, num_grad(lambda: cross_entropy(m.forward(x), y)[0], x), atol=1e-6, rtol=1e-4)


def test_attacks_stay_in_budget_and_raise_the_loss():
    m, _, Xva, yva = get_trained()
    x, y = Xva[:64], yva[:64]
    base = cross_entropy(m.predict_logits(x), y)[0]
    eps = 0.05
    for adv in (fgsm(m, x, y, eps), pgd(m, x, y, eps, steps=5)):
        assert np.abs(adv - x).max() <= eps + 1e-6
        assert adv.min() >= 0 and adv.max() <= 1
        assert cross_entropy(m.predict_logits(adv), y)[0] > base
    # PGD is at least as strong as one FGSM step on accuracy
    acc = lambda a: (m.predict(a) == y).mean()
    assert acc(pgd(m, x, y, eps, steps=10)) <= acc(fgsm(m, x, y, eps)) + 0.05


def test_adversarial_training_improves_fgsm_accuracy():
    Xtr, ytr = make_dataset(900, seed=1)
    Xte, yte = make_dataset(200, seed=5)
    kw = dict(epochs=6, log=None)
    std = build_cnn(ModelConfig(width=6, hidden=32, seed=1))
    adv = build_cnn(ModelConfig(width=6, hidden=32, seed=1))
    train(std, Xtr, ytr, **kw)
    train(adv, Xtr, ytr, adv_eps=0.05, **kw)
    a_std = (std.predict(fgsm(std, Xte, yte, 0.05)) == yte).mean()
    a_adv = (adv.predict(fgsm(adv, Xte, yte, 0.05)) == yte).mean()
    assert a_adv > a_std


def test_robustness_reports_have_expected_structure_and_monotone_noise():
    m, _, Xva, yva = get_trained()
    nat = natural_report(m, Xva, yva)
    assert set(nat) == {"noise", "shift", "contrast"}
    assert nat["noise"][0.0] >= nat["noise"][0.4]
    att = attack_report(m, Xva[:100], yva[:100], eps=(0.0, 0.1))
    assert att["fgsm"][0.0] == att["pgd"][0.0] > att["pgd"][0.1]


def test_shift_zero_fills_without_wraparound():
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    s = shift(x, 1, 1)
    np.testing.assert_array_equal(s[0, 0], [[0, 0, 0], [0, 1, 2], [0, 4, 5]])
    np.testing.assert_array_equal(shift(x, 0, 0), x)
    np.testing.assert_array_equal(shift(x, -1, 0)[0, 0][-1], [0, 0, 0])


def test_metrics():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    p, r, f = per_class(cm)
    assert np.allclose(p, [0.5, 2 / 3, 0]) and np.allclose(r, [0.5, 1, 0])
    lo, hi = wilson(90, 100)
    assert lo < 0.9 < hi and 0.82 < lo and hi < 0.96
    assert wilson(0, 0) == (0.0, 1.0)
