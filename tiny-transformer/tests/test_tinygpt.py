import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, Adam, CharTokenizer  # noqa: E402


def make(dtype=np.float64):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8)
    m = GPT(cfg, seed=1, dtype=dtype)
    rng = np.random.default_rng(2)
    # larger init so every gradient is non-trivial
    for k, v in m.params.items():
        v += rng.standard_normal(v.shape) * 0.1
    idx = rng.integers(0, 11, (2, 6))
    tgt = rng.integers(0, 11, (2, 6))
    return m, idx, tgt


def test_gradients_match_finite_differences():
    m, idx, tgt = make()
    _, grads = m.loss_and_grads(idx, tgt)
    rng = np.random.default_rng(3)
    eps = 1e-5
    for name, p in m.params.items():
        for _ in range(3):
            j = tuple(rng.integers(0, s) for s in p.shape)
            old = p[j]
            p[j] = old + eps
            lp = m.forward(idx, tgt)[1]
            p[j] = old - eps
            lm = m.forward(idx, tgt)[1]
            p[j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[name][j]) < 1e-6 + 1e-4 * abs(num), name


def test_causality():
    m, idx, _ = make()
    a = m.forward(idx)[0]
    idx2 = idx.copy()
    idx2[:, -1] = (idx2[:, -1] + 1) % 11
    b = m.forward(idx2)[0]
    assert np.allclose(a[:, :-1], b[:, :-1])
    assert not np.allclose(a[:, -1], b[:, -1])


def test_initial_loss_near_log_vocab():
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16)
    m = GPT(cfg)
    idx = np.random.default_rng(0).integers(0, 50, (4, 8))
    loss = m.forward(idx, idx)[1]
    assert abs(loss - np.log(50)) < 0.1


def test_training_overfits_small_corpus():
    text = "abcabcabcabcabcabcabcabc" * 4
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    m = GPT(GPTConfig(tok.vocab_size, 8, 1, 2, 16), seed=0)
    opt = Adam(m.params, lr=1e-2)
    x = np.stack([data[i:i + 8] for i in range(0, 16)])
    y = np.stack([data[i + 1:i + 9] for i in range(0, 16)])
    first = m.loss_and_grads(x, y)[0]
    for _ in range(60):
        loss, g = m.loss_and_grads(x, y)
        opt.step(g)
    assert loss < first * 0.2
    out = tok.decode(m.generate(tok.encode("abc"), 6, temperature=0))
    assert out == "abcabcabc"


def test_save_load_roundtrip(tmp_path):
    m, idx, _ = make(np.float32)
    path = tmp_path / "w.npz"
    m.save(path)
    m2 = GPT(m.cfg, seed=99)
    m2.load(path)
    assert np.allclose(m.forward(idx)[0], m2.forward(idx)[0])


def test_kv_cache_matches_full_forward_and_generate():
    from tinygpt.kvcache import generate_cached, _prefill
    m, idx, _ = make()
    logits, _ = _prefill(m, idx[0])
    assert np.allclose(logits, m.forward(idx[:1])[0][0, -1], atol=1e-9)
    prompt = [1, 2, 3]
    for kw in ({"temperature": 0}, {"temperature": 0.8, "top_k": 4, "seed": 5}):
        # 14 new tokens exceeds block_size=8 so the sliding window path is covered
        a = m.generate(prompt, 14, **kw)
        b = generate_cached(m, prompt, 14, **kw)
        assert list(a) == list(b)


def test_kv_cache_matches_generate_with_top_p():
    from tinygpt.kvcache import generate_cached
    m, _, _ = make()
    kw = {"temperature": 0.9, "top_k": 6, "top_p": 0.8, "seed": 3}
    assert list(m.generate([1, 2], 12, **kw)) == list(generate_cached(m, [1, 2], 12, **kw))


def test_kv_cache_matches_generate_with_repetition_penalty():
    from tinygpt.kvcache import generate_cached
    m, _, _ = make()
    kw = {"temperature": 0.9, "top_k": 6, "seed": 5, "repetition_penalty": 1.5}
    assert list(m.generate([1, 2], 14, **kw)) == list(generate_cached(m, [1, 2], 14, **kw))
