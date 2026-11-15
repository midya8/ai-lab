import numpy as np
from diffusionlab import ring, linear_schedule, oracle_eps_fn, sample, sliced_wasserstein
from diffusionlab.dpm import sample_dpm2


def test_dpm2_shapes_and_finite():
    g, s = ring(), linear_schedule(100)
    x = sample_dpm2(oracle_eps_fn(g, s), s, 64, 2, steps=5)
    assert x.shape == (64, 2) and np.isfinite(x).all()


def test_dpm2_matches_data_with_few_steps():
    g, s = ring(), linear_schedule(100)
    rng = np.random.default_rng(1)
    ref = g.sample(3000, rng)[0] if isinstance(g.sample(2, rng), tuple) else g.sample(3000, rng)
    x = sample_dpm2(oracle_eps_fn(g, s), s, 3000, 2, steps=8, rng=np.random.default_rng(2))
    assert sliced_wasserstein(x, ref) < 0.15


def test_dpm2_beats_ddim_at_equal_nfe_16():
    g, s = ring(), linear_schedule(100)
    rng = np.random.default_rng(1)
    r = g.sample(3000, rng)
    ref = r[0] if isinstance(r, tuple) else r
    xT = np.random.default_rng(5).standard_normal((3000, 2))
    f = oracle_eps_fn(g, s)
    d = sliced_wasserstein(sample(f, s, 3000, 2, steps=16, x_T=xT), ref)
    p = sliced_wasserstein(sample_dpm2(f, s, 3000, 2, steps=8, x_T=xT), ref)  # equal NFE (16)
    assert p < d
