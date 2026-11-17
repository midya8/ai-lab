import numpy as np
import pytest


@pytest.fixture
def data1d():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (40, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(40)
    return X, y
