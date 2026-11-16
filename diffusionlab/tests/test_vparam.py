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
