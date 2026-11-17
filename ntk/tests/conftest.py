import numpy as np
import pytest


@pytest.fixture
def circle():
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    return np.stack([np.cos(phi), np.sin(phi)], axis=1), phi
