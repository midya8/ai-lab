import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loralab.hub import AdapterHub, BASE, train_adapter  # noqa: E402
from loralab.domain import generate_domain_text  # noqa: E402
from loralab import hub_demo  # noqa: E402
from tinygpt import GPT, GPTConfig  # noqa: E402


def make_hub(n=2, r=2, seed=1):
    m = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8), seed=seed, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    hub = AdapterHub(m, r=r, alpha=4.0)
    for i in range(n):
        lo = hub.add(f"a{i}", seed=10 + i)
        for k in lo.names:
            lo.B[k][...] = rng.standard_normal(lo.B[k].shape) * 0.3
    idx = rng.integers(0, 11, (2, 6))
    return m, hub, idx


def logits(hub, idx, spec):
    with hub.active(spec) as m:
        return m.forward(idx)[0].copy()


def test_no_adapter_is_bit_exact_base_and_weights_restored():
    m, hub, idx = make_hub()
    before = {k: v.copy() for k, v in m.params.items()}
    base = m.forward(idx)[0].copy()
    assert np.array_equal(logits(hub, idx, None), base)
    logits(hub, idx, {"a0": 0.5, "a1": 0.5})
    assert all(np.array_equal(before[k], m.params[k]) for k in before)  # context exit restores


def test_single_adapter_matches_gptlora_and_differs_from_base():
    m, hub, idx = make_hub()
    expect = hub.adapters["a0"].forward(idx)[0]
    assert np.allclose(logits(hub, idx, "a0"), expect)
    assert not np.allclose(logits(hub, idx, "a0"), logits(hub, idx, None))


def test_blend_adds_deltas_linearly():
    m, hub, idx = make_hub()
    spec = {"a0": 0.3, "a1": 0.7}
    got = hub.blended_delta(spec)
    for n in got:
        want = 0.3 * hub.adapters["a0"].delta(n) + 0.7 * hub.adapters["a1"].delta(n)
        assert np.allclose(got[n], want)
    # a blend with weight 1 on one adapter equals activating it by name
    assert np.allclose(logits(hub, idx, {"a1": 1.0}), logits(hub, idx, "a1"))


def test_blend_is_not_the_average_of_factors():
    """mean(A) @ mean(B) differs from mean(A @ B): the hub must blend deltas."""
    m, hub, idx = make_hub()
    a, b = hub.adapters["a0"], hub.adapters["a1"]
    n = a.names[0]
    naive = a.scale * ((a.A[n] + b.A[n]) / 2) @ ((a.B[n] + b.B[n]) / 2)
    assert not np.allclose(naive, hub.blended_delta({"a0": .5, "a1": .5})[n])


def test_compose_is_exact_at_sum_of_ranks():
    m, hub, idx = make_hub(r=2)
    spec = {"a0": 0.4, "a1": 0.6}
    new, err = hub.compose(spec, "mix")  # rank defaults to 2 + 2
    assert new.r == 4 and err < 1e-10
    assert np.allclose(logits(hub, idx, "mix"), logits(hub, idx, spec), atol=1e-8)


def test_compose_truncation_reports_error_and_shrinks_with_rank():
    m, hub, idx = make_hub(r=3)
    spec = {"a0": 0.5, "a1": 0.5}
    _, e1 = hub.compose(spec, "r1", rank=1)
    _, e3 = hub.compose(spec, "r3", rank=3)
    _, e6 = hub.compose(spec, "r6", rank=6)
    assert e1 > e3 > e6 and e6 < 1e-10


def test_registry_errors_and_remove():
    m, hub, idx = make_hub()
    with pytest.raises(ValueError):
        hub.add("a0")
    with pytest.raises(ValueError):
        hub.add(BASE)
    with pytest.raises(KeyError):
        logits(hub, idx, "nope")
    hub.remove("a1")
    assert hub.names() == ["a0"]
    with pytest.raises(ValueError):
        hub.compose({}, "x")


def test_merged_adapter_is_refused():
    m, hub, idx = make_hub()
    hub.adapters["a0"].merge()
    with pytest.raises(RuntimeError):
        logits(hub, idx, "a0")
    hub.adapters["a0"].unmerge()


def test_hub_save_load_roundtrip(tmp_path):
    m, hub, idx = make_hub()
    hub.compose({"a0": .5, "a1": .5}, "mix", rank=3)  # mixed ranks in one hub
    hub.save(str(tmp_path))
    hub2 = AdapterHub.load(m, str(tmp_path))
    assert hub2.names() == hub.names() and hub2.targets == hub.targets
    for name in hub.names():
        assert np.allclose(logits(hub2, idx, name), logits(hub, idx, name))


def test_prompt_nll_matches_manual_and_handles_odd_lengths():
    m, hub, idx = make_hub()
    ids = np.random.default_rng(5).integers(0, 11, 20)  # 19 targets over windows of 8: 8+8+3
    got = hub.prompt_nll(ids, "a0")
    with hub.active("a0") as mm:
        tot = 0.0
        for s in (0, 8, 16):
            x, y = ids[s:s + 8][None], ids[s + 1:s + 9][None]
            x = x[:, :y.shape[1]]
            tot += mm.forward(x, y)[1] * y.shape[1]
    assert abs(got - tot / 19) < 1e-12
    with pytest.raises(ValueError):
        hub.prompt_nll([3])


def test_domain_generator_domains():
    t, k = generate_domain_text(5, seed=1), generate_domain_text(5, seed=1, domain="kitchen")
    assert t != k and t == generate_domain_text(5, seed=1)
    with pytest.raises(ValueError):
        generate_domain_text(3, domain="nope")


def test_train_adapter_reduces_loss():
    m, hub, idx = make_hub(n=0)
    data = np.tile(np.arange(11), 30)
    lo = hub.add("t", seed=0)
    curve = train_adapter(lo, data, steps=40, block=8, lr=3e-2)
    assert np.mean(curve[-5:]) < np.mean(curve[:5])


def test_end_to_end_routing_and_blend_demo():
    o = hub_demo.run(pre_steps=120, ad_steps=100, rank=4)
    # each specialist wins its own domain among adapters, and beats base there
    assert o["nll"]["tech"]["B"] < o["nll"][BASE]["B"] and o["nll"]["tech"]["B"] < o["nll"]["kitchen"]["B"]
    assert o["nll"]["kitchen"]["C"] < o["nll"][BASE]["C"] and o["nll"]["kitchen"]["C"] < o["nll"]["tech"]["C"]
    assert o["route_accuracy"] >= 0.9
    assert o["compose_rel_error"] < 1e-5  # float32 model
    assert o["base_frozen"]
