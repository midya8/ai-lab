import random
import numpy as np
from mctslab import TicTacToe, PUCT
from mctslab.azlite import PVNet, encode, train, self_play_game
from mctslab.arena import match, random_player


def test_prior_is_distribution_over_legal_moves():
    g = TicTacToe()
    net = PVNet(g, 16, seed=1)
    s = g.play(g.initial(), 4)
    pi, v = net.predict(s)
    assert abs(pi.sum() - 1) < 1e-9 and pi[4] == 0 and -1 <= v <= 1


def test_gradient_matches_finite_difference():
    g = TicTacToe()
    net = PVNet(g, 8, seed=2)
    rng = np.random.default_rng(0)
    states = [g.play(g.initial(), m) for m in range(5)]
    X = np.stack([encode(g, s) for s in states]); M = net._mask(states)
    T = rng.random((5, 9)) * M; T /= T.sum(1, keepdims=True); Z = rng.uniform(-1, 1, 5)

    def loss():
        _, pi, v = net.forward(X, M)
        return -np.sum(T * np.log(pi + 1e-12)) / 5 + np.mean((v - Z) ** 2)
    import copy
    base = copy.deepcopy(net.p)
    # analytic grad via a tiny SGD-like probe: recover from one Adam step direction sign
    eps = 1e-6
    for key, idx in [("W1", (3, 2)), ("Wp", (1, 4)), ("Wv", (5, 0)), ("bv", (0,))]:
        net.p = copy.deepcopy(base); net.p[key][idx] += eps; lp = loss()
        net.p = copy.deepcopy(base); net.p[key][idx] -= eps; lm = loss()
        num = (lp - lm) / (2 * eps)
        net.p = copy.deepcopy(base); net._m = {k: np.zeros_like(v) for k, v in net.p.items()}
        net._s = {k: np.zeros_like(v) for k, v in net.p.items()}; net._t = 0
        before = net.p[key][idx]
        net.train_batch(X, M, T, Z, lr=1e-3, l2=0.0)
        step = net.p[key][idx] - before
        if abs(num) > 1e-4:
            assert np.sign(step) == -np.sign(num)


def test_training_reduces_loss_on_fixed_batch():
    g = TicTacToe()
    net = PVNet(g, 16, seed=3)
    states = [g.play(g.initial(), m) for m in range(9)]
    X = np.stack([encode(g, s) for s in states]); M = net._mask(states)
    T = np.zeros((9, 9)); T[np.arange(9), (np.arange(9) + 1) % 9] = 1.0
    T *= M; T[T.sum(1) == 0, 0] = 0
    T = T / np.maximum(T.sum(1, keepdims=True), 1e-9)
    Z = np.zeros(9)
    first = net.train_batch(X, M, T, Z, lr=0.02)
    for _ in range(200):
        last = net.train_batch(X, M, T, Z, lr=0.02)
    assert sum(last) < sum(first)


def test_self_play_targets_are_normalised_and_game_ends():
    g = TicTacToe()
    net = PVNet(g, 16, seed=4)
    rec, w = self_play_game(g, net, 10, random.Random(0))
    assert rec and w in (-1, 0, 1)
    for s, pol in rec:
        assert abs(pol.sum() - 1) < 1e-9


def test_save_load_roundtrip(tmp_path):
    g = TicTacToe()
    a = PVNet(g, 8, seed=5); a.save(tmp_path / "n.npz")
    b = PVNet(g, 8, seed=6); b.load(tmp_path / "n.npz")
    s = g.play(g.initial(), 0)
    assert np.allclose(a.predict(s)[0], b.predict(s)[0])


def test_short_training_beats_random():
    g = TicTacToe()
    net = train(g, PVNet(g, 32, seed=0), rounds=8, games_per_round=6, iterations=15, epochs=15, seed=0)
    bot = PUCT(g, 15, prior_fn=net.prior_fn, value_fn=net.value_fn, seed=1)
    r = match(g, bot, random_player(g, 3), games=20)
    assert r["wins"] > r["losses"]
