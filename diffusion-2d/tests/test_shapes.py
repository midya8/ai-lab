import numpy as np

from diffusion2d import EpsNet, linear_schedule, net_eps_fn, sample, sliced_wasserstein, train
from diffusion2d.shapes import spiral


def test_spiral_shape_and_arms():
    x, arm = spiral(4000, np.random.default_rng(0))
    assert x.shape == (4000, 2) and set(np.unique(arm)) == {0, 1}
    assert 0.4 < x.std() < 1.5
    # the two arms are point-reflections of each other (angle offset pi) so means ~ 0
    assert np.abs(x.mean(0)).max() < 0.1


def test_spiral_training_beats_noise_baseline():
    s = linear_schedule(50)
    net = EpsNet(2, 64, 3, 0, T=50, seed=0)
    train(net, lambda n, r: (spiral(n, r)[0], None), s, steps=1500, seed=0, p_uncond=0.0)
    ref = spiral(3000, np.random.default_rng(9))[0]
    gen = sample(net_eps_fn(net), s, 3000, 2, 25, 0.0, np.random.default_rng(1))
    noise = np.random.default_rng(2).standard_normal((3000, 2))
    assert sliced_wasserstein(gen, ref) < 0.5 * sliced_wasserstein(noise, ref)
