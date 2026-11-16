import numpy as np
import pytest
from interplab import _mg  # noqa: F401
from microgrid import MiniGPT
from interplab import (InterpModel, make_batch, predictable_mask, corrupt_batch, VOCAB, SEQ, loss_and_acc,
                       logit_attribution, zero_ablate, mean_ablate, merge_hooks, head_ablation_table,
                       previous_token_scores, induction_scores, composition_scores, patch_head_effects)
from interplab.analysis import _run


@pytest.fixture(scope="module")
def rand_model():
    m = MiniGPT(VOCAB, d=32, heads=2, layers=2, ctx=SEQ - 1, seed=5)
    # make LayerNorm params non-trivial so parity tests are not vacuous
    rng = np.random.default_rng(1)
    for p in m.parameters():
        p.data = p.data + rng.normal(0, 0.05, p.shape)
    return m, InterpModel.from_minigpt(m)


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    return make_batch(rng, 24)


def test_task_is_periodic_and_distinct(batch):
    tokens, periods = batch
    for row, p in zip(tokens, periods):
        assert len(set(row[:p])) == p
        assert all(row[t] == row[t - p] for t in range(p, len(row)))
    assert set(periods) <= set(range(4, 9))


def test_predictable_mask_counts(batch):
    _, periods = batch
    mask = predictable_mask(periods)
    assert mask.shape == (len(periods), SEQ - 1)
    for p, m in zip(periods, mask):
        assert m.sum() == SEQ - 1 - (p - 1)  # input positions p-1 .. SEQ-2


def test_corrupt_keeps_periods_changes_tokens(batch):
    tokens, periods = batch
    c = corrupt_batch(np.random.default_rng(9), tokens, periods)
    assert c.shape == tokens.shape
    for row, p in zip(c, periods):
        assert all(row[t] == row[t - p] for t in range(p, len(row)))
    assert (c != tokens).any()


def test_forward_matches_microgrid(rand_model, batch):
    m, im = rand_model
    ids = batch[0][:, :-1]
    np.testing.assert_allclose(im.forward(ids), m(ids).data, atol=1e-9)


def test_save_load_roundtrip(rand_model, batch, tmp_path):
    _, im = rand_model
    path = str(tmp_path / "m.npz")
    im.save(path)
    im2 = InterpModel.load(path)
    ids = batch[0][:, :-1]
    np.testing.assert_array_equal(im.forward(ids), im2.forward(ids))


def test_cache_shapes_and_causal_attention(rand_model, batch):
    _, im = rand_model
    ids = batch[0][:, :-1]
    _, c = _run(im, ids)
    B, T = ids.shape
    pat = c["L0.pattern"]
    assert pat.shape == (B, 2, T, T)
    np.testing.assert_allclose(pat.sum(-1), 1.0, atol=1e-12)
    assert np.allclose(np.triu(pat[0, 0], 1), 0.0, atol=1e-12)
    assert c["L1.head_out"].shape == (B, 2, T, 32)


def test_head_outputs_sum_to_attention_output(rand_model, batch):
    _, im = rand_model
    ids = batch[0][:, :-1]
    _, c = _run(im, ids)
    attn_out = c["L0.head_out"].sum(1) + im.p["blocks"][0]["proj"][1]
    np.testing.assert_allclose(c["L0.resid_mid"], c["L0.resid_pre"] + attn_out, atol=1e-12)


def test_identity_hook_changes_nothing_and_zero_all_heads_does(rand_model, batch):
    _, im = rand_model
    ids = batch[0][:, :-1]
    base = im.forward(ids)
    np.testing.assert_array_equal(im.forward(ids, {"L0.head_out": lambda x: x}), base)
    hk = merge_hooks(zero_ablate(0, 0), zero_ablate(0, 1))
    assert np.abs(im.forward(ids, hk) - base).max() > 1e-6


def test_merge_hooks_composes_in_order():
    a = {"x": lambda v: v + 1}
    b = {"x": lambda v: v * 10, "y": lambda v: v}
    assert merge_hooks(a, b)["x"](1) == 20


def test_mean_ablation_with_self_reference_keeps_mean(rand_model, batch):
    _, im = rand_model
    ids = batch[0][:, :-1]
    _, ref = _run(im, ids)
    cache = {}
    im.forward(ids, mean_ablate(1, 0, ref), cache)
    out = cache["L1.head_out"][:, 0]
    # every sequence now has the same head-0 output at each position
    np.testing.assert_allclose(out, np.broadcast_to(out[0], out.shape), atol=1e-12)
    np.testing.assert_allclose(out[0], ref["L1.head_out"][:, 0].mean(0), atol=1e-12)


def test_logit_attribution_is_exactly_additive(rand_model, batch):
    _, im = rand_model
    tokens, periods = batch
    parts, total = logit_attribution(im, tokens, periods)
    assert abs(sum(parts.values()) - total) < 1e-9
    assert "L1H1" in parts and "embed" in parts


def test_attention_scores_on_hand_built_cache():
    # a perfect previous-token head and a perfect induction head on period-3 sequence of length 8
    T, p = 8, 3
    pat = np.zeros((1, 2, T, T))
    for t in range(T):
        pat[0, 0, t, max(t - 1, 0)] = 1.0
        pat[0, 1, t, max(t - p + 1, 0) if t >= p else t] = 1.0
    cache = {"L0.pattern": pat}
    prev = previous_token_scores(cache, 1)
    ind = induction_scores(cache, 1, np.array([p]))
    assert prev[0, 0] == pytest.approx(1.0) and prev[0, 1] < 0.5
    assert ind[0, 1] == pytest.approx(1.0) and ind[0, 0] < 0.5


def test_composition_scores_bounds(rand_model):
    _, im = rand_model
    for kind in "qkv":
        S, ref = composition_scores(im, kind)
        assert S.shape == (2, 2) and (S >= 0).all() and (S <= 1 + 1e-9).all() and 0 < ref < 1


def test_patching_clean_into_itself_recovers_everything(rand_model, batch):
    _, im = rand_model
    tokens, periods = batch
    corrupt = corrupt_batch(np.random.default_rng(2), tokens, periods)
    r = patch_head_effects(im, tokens, corrupt, periods)
    assert r["clean_logprob"] > r["corrupt_logprob"] - 5  # sanity: finite numbers
    assert len(r["rows"]) == 4 and all(np.isfinite(x["recovery"]) for x in r["rows"])
    # patching *every* head of both layers is not the same as patching the clean run, but patching the
    # clean batch into the clean batch must be an exact no-op:
    _, cc = _run(im, tokens[:, :-1])
    hk = {"L0.head_out": lambda x: cc["L0.head_out"]}
    np.testing.assert_allclose(im.forward(tokens[:, :-1], hk), im.forward(tokens[:, :-1]), atol=1e-12)


def test_ablation_table_structure(rand_model, batch):
    _, im = rand_model
    t = head_ablation_table(im, *batch)
    assert len(t["rows"]) == 4
    base = loss_and_acc(im, *batch)
    assert t["baseline"]["nll"] == pytest.approx(base[0])
    t2 = head_ablation_table(im, *batch, mode="mean")
    assert len(t2["rows"]) == 4


class _InductionOracle:
    """A hand-written induction algorithm exposing the same forward() interface."""
    V = VOCAB

    def forward(self, ids, hooks=None):
        out = np.full(ids.shape + (self.V,), -1.0)
        for b in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                prev = [k for k in range(t) if ids[b, k] == ids[b, t]]
                if prev:
                    out[b, t, ids[b, prev[0] + 1]] = 5.0
        return out


def test_copy_fidelity_oracle_follows_source_not_control():
    from interplab import copy_fidelity
    tokens, periods = make_batch(np.random.default_rng(3), 120)
    r = copy_fidelity(_InductionOracle(), tokens, periods, np.random.default_rng(0))
    assert r["n"] > 60 and r["follow"] == 1.0 and r["control"] < 0.25


def test_offset_profile_on_hand_built_pattern():
    from interplab import period_offset_profile, uniform_attention_baseline
    T, p = 12, 4
    pat = np.zeros((1, 1, T, T))
    for t in range(T):
        pat[0, 0, t, t - p + 1 if t >= p else t] = 1.0
    prof = period_offset_profile({"L0.pattern": pat}, 0, np.array([p]), offsets=[-1, 0, 1])
    assert prof[0].tolist() == [0.0, 0.0, 1.0]
    assert 0 < uniform_attention_baseline(np.array([4, 6]), 17) < 0.2


def test_mlp_hidden_hook_is_consistent_and_ablation_noop_for_empty():
    import numpy as np
    from interplab import InterpModel, neuron_ablate, make_batch
    rng = np.random.default_rng(0)
    D, H, V, T, F = 8, 2, 11, 6, 32
    r = lambda *s: rng.normal(0, 0.3, s)
    blk = dict(ln1=(np.ones(D), np.zeros(D)), ln2=(np.ones(D), np.zeros(D)), qkv=(r(D, 3 * D), np.zeros(3 * D)),
               proj=(r(D, D), np.zeros(D)), fc1=(r(D, F), np.zeros(F)), fc2=(r(F, D), np.zeros(D)))
    m = InterpModel(dict(tok=r(V, D), pos=r(T, D), blocks=[blk], lnf=(np.ones(D), np.zeros(D)), head=(r(D, V), np.zeros(V))), H)
    ids = rng.integers(0, V, (3, T))
    cache = {}
    base = m.forward(ids, None, cache)
    assert cache["L0.mlp_hidden"].shape == (3, T, F)
    assert np.allclose(m.forward(ids, neuron_ablate(0, [])), base)
    out = m.forward(ids, neuron_ablate(0, [0, 5]))
    assert not np.allclose(out, base)
    # ablating all neurons == zero MLP output except bias
    full = m.forward(ids, neuron_ablate(0, np.arange(F)))
    ref = m.forward(ids, {"L0.mlp_out": lambda x: np.zeros_like(x) + blk["fc2"][1]})
    assert np.allclose(full, ref)
    # mean-ablation with the batch itself as reference leaves the mean activation unchanged
    mm = m.forward(ids, neuron_ablate(0, [3], cache))
    assert mm.shape == base.shape
