import numpy as np
import pytest
from diffusionlab import ring, linear_schedule, cosine_schedule, oracle_eps_fn, sample, q_sample
from diffusionlab.dpm import sample_dpm2
from diffusionlab.vparam import (v_target, eps_from_v, v_from_eps, eps_fn_to_v_fn,
                                 v_fn_to_eps_fn, snr, loss_weights)


@pytest.mark.parametrize("mk", [linear_schedule, cosine_schedule])
def test_v_roundtrip_and_target_consistency(mk):
    s = mk(100)
    rng = np.random.default_rng(0)
    x0, eps = rng.standard_normal((50, 2)), rng.standard_normal((50, 2))
    for t in (1, 30, 99):
        xt = q_sample(x0, np.full(50, t), eps, s)
        v = v_target(x0, eps, t, s)
        assert np.allclose(v, v_from_eps(xt, eps, t, s))
        assert np.allclose(eps_from_v(xt, v, t, s), eps)


def test_v_wrapped_oracle_gives_same_samples():
    g, s = ring(), linear_schedule(100)
    f = oracle_eps_fn(g, s)
    f2 = v_fn_to_eps_fn(eps_fn_to_v_fn(f, s), s)
    xT = np.random.default_rng(3).standard_normal((200, 2))
    assert np.allclose(sample(f, s, 200, 2, 10, x_T=xT), sample(f2, s, 200, 2, 10, x_T=xT), atol=1e-8)
    assert np.allclose(sample_dpm2(f, s, 200, 2, 5, x_T=xT), sample_dpm2(f2, s, 200, 2, 5, x_T=xT), atol=1e-8)


def test_weights():
    s = linear_schedule(100)
    r = snr(s)
    assert np.isinf(r[0]) and np.all(np.diff(r[1:]) < 0)
    assert np.allclose(loss_weights(s, "v")[1:], r[1:] + 1)
    assert loss_weights(s, "min_snr", 5.0)[1:].max() == 5.0
    assert np.all(loss_weights(s, "eps") == 1)
    with pytest.raises(ValueError):
        loss_weights(s, "nope")


def test_weighted_grad_matches_finite_difference():
    from diffusionlab import EpsNet
    net = EpsNet(D=2, hidden=8, depth=2, T=100, seed=1)
    rng = np.random.default_rng(0)
    for p in net.params:
        p += 0.1 * rng.standard_normal(p.shape)
    x, tgt = rng.standard_normal((6, 2)), rng.standard_normal((6, 2))
    t, w = rng.integers(1, 100, 6), rng.random(6) + 0.5
    _, grads = net.loss_and_grads(x, t, None, tgt, weights=w)
    p = net.params[2]
    for idx in [(0, 0), (3, 2)]:
        old = p[idx]
        p[idx] = old + 1e-6; lp = net.loss_and_grads(x, t, None, tgt, weights=w)[0]
        p[idx] = old - 1e-6; lm = net.loss_and_grads(x, t, None, tgt, weights=w)[0]
        p[idx] = old
        assert abs((lp - lm) / 2e-6 - grads[2][idx]) < 1e-6


def test_unit_weights_unchanged():
    from diffusionlab import EpsNet
    net = EpsNet(D=2, hidden=8, depth=2, T=100, seed=1)
    x, tgt, t = np.random.default_rng(0).standard_normal((5, 2)), np.ones((5, 2)), np.arange(1, 6)
    assert net.loss_and_grads(x, t, None, tgt)[0] == net.loss_and_grads(x, t, None, tgt, weights=np.ones(5))[0]


def test_train_param_v_and_eps_learn_ring():
    from diffusionlab import EpsNet, sliced_wasserstein
    from diffusionlab.vparam import train_param
    g, s = ring(), linear_schedule(50)
    sf = lambda n, rng: (g.sample(n, rng), None)
    ref = g.sample(1000, np.random.default_rng(9))
    res = {}
    for tg in ("eps", "v"):
        net = EpsNet(D=2, hidden=64, depth=3, T=50, seed=0)
        h = train_param(net, sf, s, target=tg, weighting="v" if tg == "v" else "eps", steps=600, batch=128)
        assert np.mean(h[-50:]) < np.mean(h[:50])
        fn = lambda x, t, l, net=net: net.predict(x, np.full(len(x), t))
        if tg == "v":
            fn = v_fn_to_eps_fn(fn, s)
        xs = sample(fn, s, 1000, 2, 25, rng=np.random.default_rng(1))
        res[tg] = sliced_wasserstein(xs, ref)
    noise = sliced_wasserstein(np.random.default_rng(2).standard_normal((1000, 2)), ref)
    assert all(v < 0.9 * noise for v in res.values()), (res, noise)


def test_train_param_bad_target():
    from diffusionlab import EpsNet
    from diffusionlab.vparam import train_param
    with pytest.raises(ValueError):
        train_param(EpsNet(T=10), None, linear_schedule(10), target="x0")
