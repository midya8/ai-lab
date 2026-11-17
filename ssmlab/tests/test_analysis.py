import numpy as np

from ssmlab import Mamba, selective_copy
from ssmlab.analysis import auc, delta_traces, force_delta_zero, gate_auc, skip_ablation


def test_auc_known_values():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(size=400)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert abs(auc(a, b) - brute) < 1e-12


def test_traces_shape_and_positive():
    m = Mamba(vocab=6, d_model=8, d_state=4, n_layer=2, seed=0)
    toks, _ = selective_copy(5, 10, 3, 4, np.random.default_rng(0))
    tr = delta_traces(m, toks)
    assert len(tr) == 2 and tr[0].shape == toks.shape and (tr[0] > 0).all()


def test_gate_auc_on_planted_signal():
    toks, _ = selective_copy(40, 12, 3, 4, np.random.default_rng(1))
    planted = np.where(toks > 0, 2.0, 0.1)
    row = gate_auc([planted], toks, 12)[0]
    assert row["auc"] == 1.0 and row["delta_data"] > row["delta_noise"]


def test_force_zero_skips_token_and_restores():
    m = Mamba(vocab=6, d_model=8, d_state=4, n_layer=2, seed=0)
    toks, _ = selective_copy(3, 8, 2, 4, np.random.default_rng(2))
    base = m.forward(toks)
    mask = np.zeros(toks.shape, bool)
    mask[:, 3] = True
    with force_delta_zero(m, mask):
        out = m.forward(toks)
        assert (m.blocks[0].ssm.cache[2][:, 3] == 0).all()
    assert not np.allclose(out, base)
    assert np.allclose(m.forward(toks), base)


def test_zero_delta_leaves_state_unchanged():
    m = Mamba(vocab=6, d_model=8, d_state=4, n_layer=1, seed=0)
    toks, _ = selective_copy(2, 8, 2, 4, np.random.default_rng(3))
    with force_delta_zero(m, np.ones(toks.shape, bool)):
        m.forward(toks)
        assert np.allclose(m.blocks[0].ssm.last_state, 0.0)


def test_skip_ablation_keys():
    m = Mamba(vocab=6, d_model=8, d_state=4, n_layer=2, seed=0)
    toks, tg = selective_copy(10, 12, 3, 4, np.random.default_rng(4))
    r = skip_ablation(m, toks, tg, 12, np.random.default_rng(0))
    assert set(r) == {"intact", "skip_data", "skip_random_noise"}
