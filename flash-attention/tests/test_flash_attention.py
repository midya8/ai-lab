import itertools

import numpy as np
import pytest

from flash_attention import (OutOfBlocks, PagedKVCache, alibi_slopes, allowed_mask, attend_partial,
                      finalize, flash_backward, flash_decode, flash_forward, merge_partials,
                      naive_attention, naive_backward, naive_traffic, split_ranges, tile_status)


def rand(H=4, Hkv=2, nq=37, nk=53, d=8, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(H, nq, d)), r.normal(size=(Hkv, nk, d)), r.normal(size=(Hkv, nk, d))


CONFIGS = [dict(), dict(causal=True), dict(causal=True, window=7), dict(window=5),
           dict(causal=True, alibi=True), dict(window=9, alibi=True)]


def kw(cfg, H):
    c = dict(cfg)
    if c.pop("alibi", False):
        c["alibi_slopes"] = alibi_slopes(H)
    return c


# ---- masks -----------------------------------------------------------------------------------
@pytest.mark.parametrize("causal,window", [(False, None), (True, None), (True, 5), (False, 5),
                                           (True, 1), (False, 1)])
def test_tile_status_matches_bruteforce(causal, window):
    off = 11
    for i0, i1, j0, j1 in itertools.product([0, 3, 9], [4, 10, 16], [0, 5, 14, 30], [6, 15, 31]):
        if i1 <= i0 or j1 <= j0:
            continue
        m = allowed_mask(i1 - i0, j1 - j0, off + i0, causal, window, k_start=j0)
        st = tile_status(i0, i1, j0, j1, off, causal, window)
        expect = "skip" if not m.any() else ("full" if m.all() else "partial")
        assert st == expect, (i0, i1, j0, j1)


def test_alibi_slopes_geometric():
    s = alibi_slopes(8)
    assert np.allclose(s, [2.0 ** (-(h + 1)) for h in range(8)])


# ---- forward ---------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("blocks", [(1, 1), (8, 16), (64, 64), (5, 7)])
def test_flash_forward_matches_naive(cfg, blocks):
    q, k, v = rand()
    c = kw(cfg, 4)
    o_ref, l_ref, _ = naive_attention(q, k, v, **c)
    o, l, st = flash_forward(q, k, v, block_q=blocks[0], block_k=blocks[1], **c)
    assert np.allclose(o, o_ref, atol=1e-12)
    ok = np.isfinite(l_ref)
    assert np.array_equal(ok, np.isfinite(l)) and np.allclose(l[ok], l_ref[ok], atol=1e-12)
    assert st.tiles_total == st.tiles_skipped + st.tiles_full + st.tiles_partial


def test_fully_masked_rows_give_zero_and_neg_inf():
    q, k, v = rand(nq=6, nk=6)
    # window=1 non-causal with q_offset far past the keys: nothing visible for any row
    o, l, _ = flash_forward(q, k, v, window=1, q_offset=100, block_q=4, block_k=4)
    assert np.all(o == 0) and np.all(np.isneginf(l))
    o_ref, l_ref, _ = naive_attention(q, k, v, window=1, q_offset=100)
    assert np.all(o_ref == 0) and np.all(np.isneginf(l_ref))


def test_softmax_rows_sum_to_one_in_reference():
    q, k, v = rand()
    _, _, p = naive_attention(q, k, v, causal=True)
    assert np.allclose(p.sum(-1), 1.0)


def test_online_softmax_survives_logits_that_overflow_naive_exp():
    q, k, v = rand(nq=8, nk=40)
    q = q * 400.0        # scores ~ hundreds: exp(score) overflows float64 only at ~709, float32 at ~88
    o32, l32, _ = flash_forward(q, k, v, dtype=np.float32, block_q=4, block_k=8)
    o_ref, _, _ = naive_attention(q, k, v)
    assert np.all(np.isfinite(o32))
    assert np.abs(o32 - o_ref).max() < 1e-3
    with np.errstate(over="ignore", invalid="ignore"):
        s = (q[0] @ k[0].T * q.shape[-1] ** -0.5).astype(np.float32)
        unsafe = np.exp(s)
    assert not np.all(np.isfinite(unsafe))     # the naive un-stabilised softmax would break


def test_float32_error_is_small_and_flash_not_worse_than_naive_scale():
    q, k, v = rand(nq=64, nk=256)
    o_ref, _, _ = naive_attention(q, k, v, causal=True)
    o32, _, _ = flash_forward(q, k, v, causal=True, dtype=np.float32, block_q=16, block_k=32)
    assert np.abs(o32 - o_ref).max() < 5e-5


def test_gqa_equals_repeated_kv_mha():
    q, k, v = rand(H=6, Hkv=2)
    o_g, _, _ = flash_forward(q, k, v, causal=True, block_q=8, block_k=8)
    o_m, _, _ = flash_forward(q, np.repeat(k, 3, axis=0), np.repeat(v, 3, axis=0), causal=True,
                              block_q=8, block_k=8)
    assert np.allclose(o_g, o_m, atol=1e-13)


def test_bad_group_raises():
    q, k, v = rand(H=5, Hkv=2)
    with pytest.raises(ValueError):
        flash_forward(q, k, v)


# ---- tile skipping and traffic ---------------------------------------------------------------
def test_causal_skips_about_half_the_tiles():
    q, k, v = rand(H=1, Hkv=1, nq=256, nk=256, d=4)
    _, _, st = flash_forward(q, k, v, causal=True, block_q=16, block_k=16)
    assert st.tiles_total == 256
    assert st.tiles_skipped == 16 * 15 // 2          # strictly-upper tiles
    assert st.tiles_partial == 16                    # the diagonal tiles
    _, _, full = flash_forward(q, k, v, block_q=16, block_k=16)
    tile_flops = 4 * 16 * 16 * 4
    assert st.flops == (16 * 15 // 2 * 0 + 120 + 16) * tile_flops    # 120 full + 16 diagonal
    assert full.flops == 256 * tile_flops


def test_window_skips_far_tiles_and_flops_scale_linearly():
    d = 4
    flops = []
    for n in (128, 256, 512):
        q, k, v = rand(H=1, Hkv=1, nq=n, nk=n, d=d)
        _, _, st = flash_forward(q, k, v, causal=True, window=32, block_q=16, block_k=16)
        flops.append(st.flops)
    assert 1.8 < flops[1] / flops[0] < 2.2 and 1.8 < flops[2] / flops[1] < 2.2


def test_unmasked_traffic_matches_closed_form():
    H, nq, nk, d, bq, bk = 3, 100, 130, 8, 16, 32
    q, k, v = rand(H=H, Hkv=H, nq=nq, nk=nk, d=d)
    _, _, st = flash_forward(q, k, v, block_q=bq, block_k=bk)
    n_qt = -(-nq // bq)
    assert st.reads == H * (nq * d + n_qt * 2 * nk * d)
    assert st.writes == H * (nq * d + nq)
    assert st.peak_tile_elems == bq * bk


def test_flash_traffic_beats_naive_when_tiles_are_large_enough():
    H, n, d = 1, 512, 16
    q, k, v = rand(H=H, Hkv=H, nq=n, nk=n, d=d)
    _, _, st = flash_forward(q, k, v, block_q=128, block_k=128)
    assert st.traffic < naive_traffic(n, n, d, H)["total"]


# ---- backward --------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS)
def test_flash_backward_matches_naive_backward(cfg):
    q, k, v = rand(nq=29, nk=41)
    c = kw(cfg, 4)
    do = np.random.default_rng(5).normal(size=q.shape)
    o, l, _ = flash_forward(q, k, v, block_q=8, block_k=8, **c)
    dq, dk, dv = flash_backward(q, k, v, o, l, do, block_q=6, block_k=10, **c)
    rq, rk, rv = naive_backward(q, k, v, do, **c)
    for a, b in ((dq, rq), (dk, rk), (dv, rv)):
        assert np.allclose(a, b, atol=1e-10)


def test_naive_backward_matches_finite_differences():
    q, k, v = rand(H=2, Hkv=1, nq=5, nk=6, d=3)
    do = np.random.default_rng(1).normal(size=q.shape)
    c = dict(causal=True, alibi_slopes=alibi_slopes(2))

    def loss(q_, k_, v_):
        return float((do * naive_attention(q_, k_, v_, **c)[0]).sum())

    grads = naive_backward(q, k, v, do, **c)
    for arr_i, arr in enumerate((q, k, v)):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            for sgn in (1, -1):
                a = arr.copy()
                a[idx] += sgn * 1e-6
                args = [q, k, v]
                args[arr_i] = a
                num[idx] += sgn * loss(*args) / 2e-6
        assert np.allclose(grads[arr_i], num, atol=1e-6)


def test_backward_with_fully_masked_rows_is_finite():
    q, k, v = rand(nq=6, nk=6)
    do = np.ones_like(q)
    o, l, _ = flash_forward(q, k, v, window=1, q_offset=100)
    dq, dk, dv = flash_backward(q, k, v, o, l, do, window=1, q_offset=100)
    assert all(np.all(g == 0) for g in (dq, dk, dv))


# ---- split-KV decoding and partial merging ----------------------------------------------------
def test_split_ranges_cover_exactly():
    for nk, s in [(10, 3), (7, 7), (7, 20), (1, 4), (100, 8)]:
        r = split_ranges(nk, s)
        assert r[0][0] == 0 and r[-1][1] == nk
        assert all(a[1] == b[0] for a, b in zip(r, r[1:]))


@pytest.mark.parametrize("splits", [1, 2, 5, 16])
@pytest.mark.parametrize("cfg", [dict(causal=True), dict(causal=True, window=11),
                                 dict(causal=True, alibi=True)])
def test_flash_decode_equals_full_attention(splits, cfg):
    q, k, v = rand(H=4, Hkv=2, nq=1, nk=90)
    c = kw(cfg, 4)
    o_ref, l_ref, _ = naive_attention(q, k, v, **c)
    o, l, st = flash_decode(q, k, v, num_splits=splits, block_k=8, **c)
    assert np.allclose(o, o_ref, atol=1e-12) and np.allclose(l, l_ref, atol=1e-12)
    assert len(st) == splits


def test_merge_is_order_and_grouping_independent():
    q, k, v = rand(nq=5, nk=60)
    parts = [attend_partial(q, k[:, a:b], v[:, a:b], q_offset=55, k_start=a, causal=True,
                            block_k=7) for a, b in split_ranges(60, 5)]
    o1, l1 = finalize(*merge_partials(parts))
    o2, l2 = finalize(*merge_partials(parts[::-1]))
    left = merge_partials(parts[:2])
    o3, l3 = finalize(*merge_partials([left] + parts[2:]))
    o_ref, l_ref, _ = naive_attention(q, k, v, causal=True)
    for o, l in ((o1, l1), (o2, l2), (o3, l3)):
        assert np.allclose(o, o_ref, atol=1e-12) and np.allclose(l, l_ref, atol=1e-12)


def test_merge_handles_slice_with_nothing_visible():
    q, k, v = rand(nq=1, nk=40)
    # window=4 at the last position: only keys 36..39 visible, so splits 0..2 are fully masked
    o, l, _ = flash_decode(q, k, v, num_splits=4, window=4, causal=True, block_k=5)
    o_ref, l_ref, _ = naive_attention(q, k, v, window=4, causal=True)
    assert np.allclose(o, o_ref, atol=1e-12) and np.allclose(l, l_ref, atol=1e-12)


# ---- paged cache -----------------------------------------------------------------------------
def make_cache(blocks=16, bs=4, Hkv=2, d=8):
    return PagedKVCache(blocks, bs, Hkv, d)


def toks(T, Hkv=2, d=8, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(T, Hkv, d)), r.normal(size=(T, Hkv, d))


def test_append_gather_roundtrip_across_block_boundaries():
    c = make_cache()
    s = c.new_sequence()
    k1, v1 = toks(5, seed=1)
    k2, v2 = toks(6, seed=2)
    c.append(s, k1, v1)
    c.append(s, k2, v2)
    K, V = c.gather(s)
    assert K.shape == (2, 11, 8)
    assert np.allclose(K.transpose(1, 0, 2), np.concatenate([k1, k2]))
    assert np.allclose(V.transpose(1, 0, 2), np.concatenate([v1, v2]))
    assert len(c.tables[s]) == 3


@pytest.mark.parametrize("cfg", [dict(causal=True), dict(causal=True, window=6),
                                 dict(causal=True, alibi=True)])
def test_paged_attention_equals_contiguous(cfg):
    c = make_cache(bs=4)
    s = c.new_sequence()
    k, v = toks(23, seed=3)
    c.append(s, k, v)
    q = np.random.default_rng(4).normal(size=(4, 3, 8))     # last 3 tokens, 4 q heads / 2 kv heads
    kw_ = kw(cfg, 4)
    o, l = c.attend(s, q, **kw_)
    K, V = c.gather(s)
    o_ref, l_ref, _ = naive_attention(q, K, V, **kw_)
    assert np.allclose(o, o_ref, atol=1e-12) and np.allclose(l, l_ref, atol=1e-12)


def test_fork_shares_blocks_and_copy_on_write_isolates_children():
    c = make_cache(blocks=10, bs=4)
    a = c.new_sequence()
    k, v = toks(10, seed=5)                       # 2 full blocks + 1 partial
    c.append(a, k, v)
    before = c.gather(a)
    used = c.blocks_in_use()
    b = c.fork(a)
    assert c.blocks_in_use() == used                       # nothing copied yet
    kn, vn = toks(1, seed=6)
    c.append(b, kn, vn)                                    # lands in the shared partial block
    assert c.copies == 1
    assert c.blocks_in_use() == used + 1
    assert np.allclose(c.gather(a)[0], before[0]) and c.lengths[a] == 10   # parent untouched
    assert np.allclose(c.gather(b)[0][:, :10], before[0])
    assert np.allclose(c.gather(b)[0][:, 10], kn[0])
    # appending to the parent now needs no copy: its tail block is exclusively its own again
    c.append(a, kn, vn)
    assert c.copies == 1


def test_fork_at_block_boundary_needs_no_copy():
    c = make_cache(bs=4)
    a = c.new_sequence()
    c.append(a, *toks(8))
    b = c.fork(a)
    c.append(b, *toks(1, seed=9))
    assert c.copies == 0


def test_free_returns_blocks_only_when_unshared():
    c = make_cache(blocks=8, bs=4)
    a = c.new_sequence()
    c.append(a, *toks(8))
    b = c.fork(a)
    c.free(a)
    assert c.blocks_in_use() == 2                          # still held by the fork
    assert np.allclose(c.gather(b)[0].shape, (2, 8, 8))
    c.free(b)
    assert c.blocks_in_use() == 0 and len(c.free_blocks) == 8


def test_out_of_blocks_is_clean_and_leaves_state_unchanged():
    c = make_cache(blocks=2, bs=4)
    a = c.new_sequence()
    c.append(a, *toks(7))
    with pytest.raises(OutOfBlocks):
        c.append(a, *toks(5))
    assert c.lengths[a] == 7 and c.blocks_in_use() == 2


def test_paged_fragmentation_beats_contiguous_reservation():
    c = PagedKVCache(64, 16, 2, 4)
    lens = [5, 37, 18, 90, 3]
    for i, n in enumerate(lens):
        s = c.new_sequence()
        c.append(s, *toks(n, d=4, seed=i))
    u = c.utilisation()
    assert u["logical_tokens"] == sum(lens)
    assert u["internal_waste_slots"] < 16 * len(lens)       # < one block wasted per sequence
    assert u["physical_slots"] < len(lens) * 128            # vs reserving max_len=128 each


def test_shared_prefix_uses_fewer_physical_slots_than_logical_tokens():
    c = make_cache(blocks=32, bs=4)
    root = c.new_sequence()
    c.append(root, *toks(16))
    kids = [c.fork(root) for _ in range(4)]
    for i, s in enumerate(kids):
        c.append(s, *toks(3, seed=10 + i))
    u = c.utilisation()
    assert u["logical_tokens"] == 16 + 4 * 19
    assert u["physical_slots"] < u["logical_tokens"]
