import numpy as np
from microgrid import Tensor, MLP, Adam, SGD, mse_loss, softmax_cross_entropy, gradcheck

rng = np.random.default_rng(1)
T = lambda *s: Tensor(rng.normal(size=s))

def chk(fn, *ins):
    assert gradcheck(fn, ins) < 1e-5

def test_arith_broadcast():
    chk(lambda a, b: ((a * b + a / (b * b + 2.0) - 1.0) ** 2).sum(), T(3, 4), T(4))

def test_matmul_2d():
    chk(lambda a, b: (a @ b).tanh().sum(), T(3, 4), T(4, 2))

def test_matmul_vec_cases():
    chk(lambda a, b: (a @ b).sum(), T(3, 4), T(4))
    chk(lambda a, b: (a @ b).sum(), T(4), T(4, 3))
    chk(lambda a, b: (a @ b) * 1.0, T(5), T(5))

def test_batched_matmul():
    chk(lambda a, b: (a @ b).sum(), T(2, 3, 4), T(4, 2))

def test_unary():
    chk(lambda a: (a.exp() + (a * a + 1).log() + a.sigmoid() + a.relu()).sum(), T(3, 3))

def test_reductions_reshape():
    chk(lambda a: (a.mean(axis=1) ** 2).sum() + a.reshape(6, 2).sum(axis=0).sum(), T(4, 3))

def test_log_softmax_ce():
    labels = np.array([0, 2, 1])
    chk(lambda x: softmax_cross_entropy(x, labels), T(3, 4))

def test_shared_node_accumulates():
    a = Tensor([2.0]); (a * a * a).sum().backward()
    assert np.allclose(a.grad, [12.0])

def test_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); y = np.array([[0], [1], [1], [0]], float)
    m = MLP([2, 8, 1], seed=3); opt = Adam(m.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad(); l = mse_loss(m(Tensor(X)), y); l.backward(); opt.step()
    assert l.data < 0.01

def test_sgd_momentum_classifier():
    r = np.random.default_rng(0)
    X = np.vstack([r.normal(-2, 1, (50, 2)), r.normal(2, 1, (50, 2))]); y = np.array([0]*50 + [1]*50)
    m = MLP([2, 6, 2], seed=1); opt = SGD(m.parameters(), lr=0.1, momentum=0.9)
    for _ in range(100):
        opt.zero_grad(); softmax_cross_entropy(m(Tensor(X)), y).backward(); opt.step()
    assert (m(Tensor(X)).data.argmax(1) == y).mean() > 0.97
