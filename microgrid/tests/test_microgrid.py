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


def test_deep_graph_no_recursion_error():
    x = Tensor(np.array([1.0]))
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert abs(x.grad[0] - 1.0001 ** 5000) < 1e-6 * 1.0001 ** 5000


def test_conv2d_grads():
    from microgrid import conv2d
    chk(lambda x, w, b: (conv2d(x, w, b).tanh()).sum(), T(2, 3, 5, 5), T(4, 3, 3, 3), T(4))


def test_conv2d_stride_pad_grads():
    from microgrid import conv2d
    chk(lambda x, w: (conv2d(x, w, stride=2, pad=1) ** 2).sum(), T(2, 2, 6, 6), T(3, 2, 3, 3))


def test_conv2d_matches_naive():
    from microgrid import conv2d
    x, w = rng.normal(size=(1, 2, 4, 4)), rng.normal(size=(2, 2, 2, 2))
    y = conv2d(Tensor(x), Tensor(w)).data
    ref = np.zeros((1, 2, 3, 3))
    for f in range(2):
        for i in range(3):
            for j in range(3):
                ref[0, f, i, j] = (x[0, :, i:i + 2, j:j + 2] * w[f]).sum()
    assert np.allclose(y, ref)


def test_max_pool_grads_and_values():
    from microgrid import max_pool2d
    x = Tensor(rng.normal(size=(2, 2, 6, 6)))
    assert max_pool2d(x).shape == (2, 2, 3, 3)
    chk(lambda a: (max_pool2d(a, 2) ** 2).sum(), T(2, 2, 6, 6))
    chk(lambda a: (max_pool2d(a, 3, stride=1) * 1.0).sum(), T(1, 2, 5, 5))


def test_tiny_cnn_learns_orientation():
    from microgrid import conv2d, max_pool2d
    r = np.random.default_rng(0)
    def make(n):
        X = r.normal(0, 0.1, (n, 1, 8, 8)); y = r.integers(0, 2, n)
        for i in range(n):
            k = r.integers(1, 7)
            if y[i]: X[i, 0, k, :] += 1.0
            else: X[i, 0, :, k] += 1.0
        return X, y
    X, y = make(120)
    w = Tensor(r.normal(0, 0.3, (4, 1, 3, 3))); b = Tensor(np.zeros(4))
    fc = Tensor(r.normal(0, 0.3, (4 * 3 * 3, 2)))
    opt = Adam([w, b, fc], lr=0.03)
    def fwd(Xb):
        h = max_pool2d(conv2d(Xb, w, b).relu(), 2)
        return h.reshape(len(Xb.data), -1) @ fc
    for _ in range(60):
        for p in (w, b, fc): p.zero_grad()
        softmax_cross_entropy(fwd(Tensor(X)), y).backward(); opt.step()
    Xt, yt = make(100)
    acc = (fwd(Tensor(Xt)).data.argmax(1) == yt).mean()
    assert acc > 0.95
