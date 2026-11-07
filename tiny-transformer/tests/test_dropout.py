import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.model import drop_fwd  # noqa: E402
from tinygpt.checkpoint import save_checkpoint, load_checkpoint  # noqa: E402
from tinygpt.tokenizer import CharTokenizer  # noqa: E402


def make(p, dtype=np.float64):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8, dropout=p)
    m = GPT(cfg, seed=1, dtype=dtype)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    return m, rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))


def test_eval_forward_ignores_dropout():
    m, idx, tgt = make(0.5)
    a = m.forward(idx, tgt)[1]
    assert a == m.forward(idx, tgt)[1]
    assert a == make(0.0)[0].forward(idx, tgt)[1]


def test_train_forward_is_stochastic_but_seed_reproducible():
    m, idx, tgt = make(0.3)
    l1 = m.forward(idx, tgt, rng=np.random.default_rng(5))[1]
    l2 = m.forward(idx, tgt, rng=np.random.default_rng(5))[1]
    l3 = m.forward(idx, tgt, rng=np.random.default_rng(6))[1]
    assert l1 == l2 and l1 != l3


def test_inverted_scaling_preserves_mean_and_rate():
    x = np.ones((200, 200))
    y, mask = drop_fwd(x, 0.25, np.random.default_rng(0))
    assert abs(y.mean() - 1.0) < 0.02
    assert abs((y == 0).mean() - 0.25) < 0.02
    assert set(np.unique(mask)) <= {0.0, 1 / 0.75}


def test_gradients_match_finite_differences_with_fixed_masks():
    m, idx, tgt = make(0.3)
    _, grads = m.loss_and_grads(idx, tgt, rng=np.random.default_rng(9))
    r = np.random.default_rng(3)
    eps = 1e-5
    for name, p in m.params.items():
        for _ in range(3):
            j = tuple(r.integers(0, s) for s in p.shape)
            old = p[j]
            p[j] = old + eps
            lp = m.forward(idx, tgt, rng=np.random.default_rng(9))[1]
            p[j] = old - eps
            lm = m.forward(idx, tgt, rng=np.random.default_rng(9))[1]
            p[j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[name][j]) < 1e-6 + 1e-4 * abs(num), name


def test_zero_dropout_with_rng_matches_plain_grads():
    m, idx, tgt = make(0.0)
    _, g1 = m.loss_and_grads(idx, tgt)
    _, g2 = m.loss_and_grads(idx, tgt, rng=np.random.default_rng(1))
    for k in g1:
        assert np.array_equal(g1[k], g2[k])


def test_checkpoint_keeps_dropout_and_old_configs_load(tmp_path):
    tok = CharTokenizer("abcdefghij")
    m = GPT(GPTConfig(tok.vocab_size, 8, 1, 2, 8, dropout=0.2))
    save_checkpoint(str(tmp_path / "c.npz"), m, tok)
    m2, _ = load_checkpoint(str(tmp_path / "c.npz"))
    assert m2.cfg.dropout == 0.2
    assert GPTConfig(vocab_size=5).dropout == 0.0


def test_training_with_dropout_still_learns():
    text = "the quick brown fox jumps over the lazy dog. " * 6
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    from tinygpt import Adam
    m = GPT(GPTConfig(tok.vocab_size, 16, 1, 2, 32, dropout=0.1), seed=0)
    opt = Adam(m.params, lr=3e-3)
    rng, drng = np.random.default_rng(0), np.random.default_rng(1)
    x = y = None
    first = None
    for step in range(120):
        ix = rng.integers(0, len(data) - 17, 8)
        x = np.stack([data[i:i + 16] for i in ix])
        y = np.stack([data[i + 1:i + 17] for i in ix])
        loss, g = m.loss_and_grads(x, y, rng=drng)
        first = loss if first is None else first
        opt.step(g)
    assert m.forward(x, y)[1] < 0.6 * first
