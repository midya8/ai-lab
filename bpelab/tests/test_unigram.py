import math
from bpelab import UnigramTokenizer
from bpelab.unigram import viterbi, _em_counts, _words

TEXT = "the quick brown fox jumps over the lazy dog. " * 30 + "unbelievably unbelievable believers happy happiness"


def test_roundtrip_lossless_incl_unseen_unicode():
    t = UnigramTokenizer.train(TEXT, 300, seed_size=500)
    for s in ["the lazy dog", "naïve café 日本語 \n\t", "zzzqqq 123"]:
        assert t.decode(t.encode(s)) == s


def test_vocab_pruned_to_budget_and_bytes_kept():
    t = UnigramTokenizer.train(TEXT, 300, seed_size=500)
    assert t.vocab_size <= 300
    assert all(bytes([i]) in t.ids for i in range(256))


def test_compresses_better_than_bytes():
    t = UnigramTokenizer.train(TEXT, 320, seed_size=500)
    assert t.compression(TEXT) > 2.0


def test_viterbi_picks_max_likelihood():
    logp = {b"a": math.log(.1), b"b": math.log(.1), b"ab": math.log(.5)}
    seg, lp = viterbi(b"abab", logp, 2)
    assert seg == [b"ab", b"ab"] and abs(lp - 2 * math.log(.5)) < 1e-12


def test_em_expected_counts_sum_to_one_per_word_when_unambiguous():
    logp = {b"a": math.log(.5), b"b": math.log(.5)}
    exp, ll = _em_counts({b"ab": 3}, logp, 1)
    assert abs(exp[b"a"] - 3) < 1e-9 and abs(ll - 3 * math.log(.25)) < 1e-9


def test_em_does_not_decrease_likelihood():
    from bpelab.unigram import _seed, _normalise
    from collections import Counter
    words = _words(TEXT)
    pieces = [bytes([i]) for i in range(256)] + _seed(words, 200)
    logp = _normalise(Counter({p: 1.0 for p in pieces}), pieces)
    ml = max(len(p) for p in pieces)
    prev = -math.inf
    for _ in range(4):
        exp, ll = _em_counts(words, logp, ml)
        assert ll >= prev - 1e-6
        prev = ll
        logp = _normalise(exp, list(logp))


def test_save_load(tmp_path):
    t = UnigramTokenizer.train(TEXT, 300, seed_size=500)
    p = tmp_path / "u.json"; t.save(p)
    u = UnigramTokenizer.load(p)
    assert u.encode(TEXT) == t.encode(TEXT) and u.vocab_size == t.vocab_size
