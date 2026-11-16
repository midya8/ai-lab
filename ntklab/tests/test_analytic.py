import numpy as np
import pytest
from ntklab import MLP, ntk, nngp


def test_kernel_is_symmetric_psd(circle):
    X, _ = circle
    K = ntk(X, depth=3)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K)[0] > -1e-9


def test_linear_net_ntk_closed_form():
    # depth 0: f = sqrt(1/d) w.x + sb b -> Theta = x.x'/d + sb2 ... plus nothing else
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    assert np.allclose(ntk(X, depth=0, sb2=0.2), X @ X.T / 3 + 0.2)


def test_depth_one_relu_ntk_matches_hand_computation():
    # one hidden layer, unit-norm 1-D-ish inputs, no bias, closed forms from arc-cos kernels
    x = np.array([[1.0, 0.0], [np.cos(0.7), np.sin(0.7)]])
    th = 0.7
    # Sigma1 = 2/d x.x' = cos(th) (d=2, |x|=1 -> diag 1)
    s1 = np.cos(th)
    e_relu = (np.sin(th) + (np.pi - th) * np.cos(th)) / (2 * np.pi)
    e_drelu = (np.pi - th) / (2 * np.pi)
    sig_out = 1.0 * e_relu
    expect = s1 * e_drelu + sig_out          # Theta1 * Sigma_dot + Sigma2   (sw2 out = 1)
    K = ntk(x, depth=1, sb2=0.0)
    assert np.isclose(K[0, 1], expect)


def test_diagonal_preserved_without_bias():
    # sw2=2 hidden keeps E h^2 = |x|^2*2/d; the output layer has sw2=1
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 6))
    n = nngp(X, depth=3, sb2=0.0)
    expect = (X * X).sum(1) * 2 / 6 / 2.0 * 1.0 * 1.0 * 1.0
    # Sigma_{L+1}(x,x) = sw2_out * E relu^2 = 1 * (2|x|^2/d)/2
    assert np.allclose(np.diag(n), (X * X).sum(1) / 6)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_empirical_ntk_converges_to_analytic_with_width(circle, depth):
    X, _ = circle
    Kinf = ntk(X, depth=depth, sb2=0.1)
    errs = {}
    for w in (32, 4096):
        e = []
        for s in range(6):
            Kh = MLP(2, w, depth, sb2=0.1, seed=100 * depth + s).ntk(X)
            e.append(np.linalg.norm(Kh - Kinf) / np.linalg.norm(Kinf))
        errs[w] = np.mean(e)
    assert errs[4096] < 0.07
    assert errs[4096] < 0.4 * errs[32]


def test_mean_of_empirical_ntk_is_unbiased_at_large_width(circle):
    X, _ = circle
    Kinf = ntk(X, depth=2)
    mean = np.mean([MLP(2, 2048, 2, seed=s).ntk(X) for s in range(10)], axis=0)
    assert np.linalg.norm(mean - Kinf) / np.linalg.norm(Kinf) < 0.03


def test_nngp_matches_function_covariance(circle):
    X, _ = circle
    F = np.stack([MLP(2, 512, 2, seed=s)(X) for s in range(1500)])
    emp = F.T @ F / len(F)
    K = nngp(X, depth=2)
    assert np.linalg.norm(emp - K) / np.linalg.norm(K) < 0.12


def test_ntk_wrong_constant_would_fail(circle):
    """Guard: using sw2=1 for hidden layers (a plausible slip) must not match the net."""
    X, _ = circle
    K = ntk(X, depth=2)
    Kh = np.mean([MLP(2, 2048, 2, seed=s).ntk(X) for s in range(4)], axis=0)
    assert np.linalg.norm(Kh - K) / np.linalg.norm(K) < 0.05
    assert np.linalg.norm(Kh - 0.5 * K) / np.linalg.norm(K) > 0.3
