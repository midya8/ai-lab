import numpy as np
import pytest
from microgrid import Tensor, gradcheck
from microgrid.regularize import dropout, Dropout


def test_eval_is_identity():
    x = Tensor(np.arange(6.0).reshape(2, 3))
    assert dropout(x, 0.5, np.random.default_rng(0), training=False) is x
    assert Dropout(0.5).eval()(x) is x


def test_scale_preserves_expectation():
    x = Tensor(np.ones((200, 200)))
    y = dropout(x, 0.3, np.random.default_rng(1)).data
    assert abs(y.mean() - 1.0) < 0.02
    assert set(np.unique(y.round(6))) == {0.0, round(1 / 0.7, 6)}
    assert abs((y == 0).mean() - 0.3) < 0.02


def test_gradcheck_with_fixed_mask():
    x = Tensor(np.random.default_rng(2).normal(size=(3, 4)))
    # fresh rng with the same seed each call -> same mask across finite differences
    fn = lambda a: (dropout(a, 0.4, np.random.default_rng(9)) ** 2).sum()
    assert gradcheck(fn, [x]) < 1e-5


def test_dropped_elements_get_zero_grad():
    x = Tensor(np.ones((50,)))
    y = dropout(x, 0.5, np.random.default_rng(4))
    y.sum().backward()
    assert np.array_equal(x.grad == 0, y.data == 0)


def test_invalid_p():
    with pytest.raises(ValueError):
        dropout(Tensor([1.0]), 1.0, np.random.default_rng(0))
