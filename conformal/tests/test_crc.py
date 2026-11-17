import numpy as np
from conformal_pred import conformal_risk_control


def _fnr_loss(lam, item):
    probs, truth = item
    pred = probs >= 1 - lam          # larger lambda -> more labels included
    return 1.0 - (pred & truth).sum() / max(truth.sum(), 1)


def _make(rng, n, k=8):
    probs = rng.random((n, k))
    truth = rng.random((n, k)) < probs ** 2
    return list(zip(probs, truth))


def test_fnr_controlled_on_average():
    lams = np.linspace(0, 1, 101)
    alpha, losses = 0.2, []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        lam, ok = conformal_risk_control(_fnr_loss, lams, _make(rng, 300), alpha)
        assert ok
        losses += [_fnr_loss(lam, d) for d in _make(rng, 300)]
    assert np.mean(losses) <= alpha + 0.02


def test_binary_loss_matches_conformal_quantile_rule():
    # 0/1 miscoverage loss: CRC reduces to the split-conformal quantile rule.
    rng = np.random.default_rng(0)
    scores = rng.random(99)
    lams = np.linspace(0, 1, 1001)
    lam, ok = conformal_risk_control(lambda l, s: float(s > l), lams, list(scores), 0.1)
    q = np.sort(scores)[int(np.ceil(100 * 0.9)) - 1]
    assert ok and q <= lam < q + 1e-3


def test_infeasible_flagged():
    lam, ok = conformal_risk_control(lambda l, d: 1.0, [0, 1], [0, 1, 2], 0.1)
    assert not ok and lam == 1
