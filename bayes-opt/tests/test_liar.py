import numpy as np
import pytest
from bayesopt import BayesOpt, branin


def _warm(liar, seed=0):
    bo = BayesOpt(branin.bounds, rng=np.random.default_rng(seed), liar=liar)
    bo.run(branin, 8)
    return bo


@pytest.mark.parametrize("liar", ["min", "mean", "max", "believer"])
def test_batch_points_are_distinct_and_in_bounds(liar):
    X = _warm(liar).ask(q=3)
    assert X.shape == (3, 2)
    assert (X >= branin.bounds[:, 0] - 1e-9).all() and (X <= branin.bounds[:, 1] + 1e-9).all()
    assert len({tuple(np.round(x, 6)) for x in X}) == 3


def test_invalid_liar_rejected():
    with pytest.raises(ValueError):
        BayesOpt(branin.bounds, liar="nope")


def test_liars_change_the_batch():
    a = _warm("min").ask(q=3)
    b = _warm("max").ask(q=3)
    assert np.allclose(a[0], b[0]) and not np.allclose(a[1:], b[1:])
