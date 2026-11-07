import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from loralab.lora import LoRALinear, MLP
from loralab.cli import run


def test_zero_init_is_identity():
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(5, 3)), rng.normal(size=3)
    x = rng.normal(size=(4, 5))
    assert np.allclose(LoRALinear(W, b, 2).forward(x), x @ W + b)


def test_gradcheck():
    rng = np.random.default_rng(1)
    l = LoRALinear(rng.normal(size=(5, 3)), np.zeros(3), 2, rng=rng)
    l.B = rng.normal(size=l.B.shape)
    x = rng.normal(size=(4, 5)); t = rng.normal(size=(4, 3))
    f = lambda: 0.5 * ((l.forward(x) - t) ** 2).sum()
    l.backward(l.forward(x) - t)
    for P, g in ((l.A, l.dA), (l.B, l.dB)):
        i = (1, 1); old = P[i]; e = 1e-6
        P[i] = old + e; a = f(); P[i] = old - e; b = f(); P[i] = old
        assert abs((a - b) / (2 * e) - g[i]) < 1e-5


def test_merge_unmerge_equivalence():
    rng = np.random.default_rng(2)
    l = LoRALinear(rng.normal(size=(5, 3)), np.zeros(3), 2, rng=rng)
    l.B = rng.normal(size=l.B.shape); x = rng.normal(size=(4, 5))
    y = l.forward(x); l.merge(); assert np.allclose(l.forward(x), y)
    l.unmerge(); assert np.allclose(l.forward(x), y)


def test_base_weights_frozen_and_adaptation_helps():
    rng = np.random.default_rng(0)
    m = MLP(8, 8, 2, rng); W1 = m.W1.copy()
    m.attach_lora(2); x = rng.normal(size=(10, 8)); y = rng.integers(0, 2, 10)
    for _ in range(5): m.step_lora(x, y, 0.1)
    assert np.array_equal(W1, m.W1)
    r = run(rank=2, steps=300)
    assert r["lora_acc_task_B"] > r["base_acc_task_B"]
    assert r["trainable_params"] < r["full_params"] / 4
    assert r["merge_preserves_predictions"]
