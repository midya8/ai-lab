import numpy as np
from microgrid import (Tensor, gradcheck, transpose, take_rows, softmax, gelu, masked_fill,
                       LayerNorm, CausalSelfAttention, MiniGPT, Adam, softmax_cross_entropy)

rng = np.random.default_rng(3)
T = lambda *s: Tensor(rng.normal(size=s))
chk = lambda fn, *ins: gradcheck(fn, ins) < 1e-5


def test_transpose_softmax_gelu():
    w = rng.normal(size=(2, 4, 3))
    assert chk(lambda a: (softmax(transpose(a, 1, 2)) * w).sum(), T(2, 3, 4))
    assert chk(lambda a: (gelu(a) * a).sum(), T(3, 4))


def test_take_rows_repeated_ids():
    assert chk(lambda t: (take_rows(t, [1, 1, 3]) ** 2).sum(), T(5, 2))


def test_masked_fill_blocks_grad():
    a = T(3, 3)
    mask = np.eye(3, dtype=bool)
    masked_fill(a, mask, 0.0).sum().backward()
    assert (a.grad[mask] == 0).all() and (a.grad[~mask] == 1).all()


def test_layernorm_stats_and_grad():
    ln = LayerNorm(6)
    y = ln(T(4, 6)).data
    assert np.allclose(y.mean(-1), 0, atol=1e-9) and np.allclose(y.var(-1), 1, atol=1e-3)
    x = T(2, 6)
    assert chk(lambda x, g, b: (ln(x) * Tensor(np.arange(6.0))).sum(), x, ln.g, ln.b)


def test_attention_is_causal():
    att = CausalSelfAttention(8, 2, np.random.default_rng(0))
    x = rng.normal(size=(1, 5, 8))
    y1 = att(Tensor(x)).data
    x2 = x.copy(); x2[0, 4] += 5.0
    y2 = att(Tensor(x2)).data
    assert np.allclose(y1[:, :4], y2[:, :4]) and not np.allclose(y1[:, 4], y2[:, 4])


def test_attention_gradcheck():
    att = CausalSelfAttention(4, 2, np.random.default_rng(1))
    w = rng.normal(size=(1, 3, 4))
    assert chk(lambda x, wq: (att(x) * w).sum(), T(1, 3, 4), att.qkv.w)


def test_minigpt_gradcheck_small():
    m = MiniGPT(5, d=4, heads=2, layers=1, ctx=4, seed=0)
    ids = [[0, 1, 2]]
    assert chk(lambda p: (m(ids) * np.arange(15.0).reshape(1, 3, 5)).sum(), m.tok)


def test_minigpt_learns_cyclic_sequence():
    V, ctx = 6, 8
    m = MiniGPT(V, d=16, heads=2, layers=1, ctx=ctx, seed=0)
    opt = Adam(m.parameters(), lr=0.02)
    seq = np.arange(ctx + 1) % V
    x, y = seq[None, :-1], seq[1:]
    for _ in range(150):
        opt.zero_grad()
        loss = softmax_cross_entropy(m(x).reshape(ctx, V), y)
        loss.backward(); opt.step()
    assert loss.data < 0.05
    assert (m(x).data[0].argmax(-1) == y).all()
