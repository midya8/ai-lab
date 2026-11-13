import random
from bpelab import BPETokenizer, train_bpe

TEXT = ("the quick brown fox jumps over the lazy dog. " * 40
        + "Tokenizers split words into subwords; naïve café 日本語 😀 123 456.\n" * 20)


def test_roundtrip_unicode():
    t = BPETokenizer.train(TEXT, 320)
    for s in ["", "hello world", "naïve café 日本語 😀", "  two  spaces\n\ttab", "don't stop"]:
        assert t.decode(t.encode(s)) == s


def test_roundtrip_random_bytes_strings():
    t = BPETokenizer.train(TEXT, 300)
    rnd = random.Random(0)
    for _ in range(50):
        s = "".join(chr(rnd.choice([rnd.randrange(32, 127), rnd.randrange(0x100, 0x2000)])) for _ in range(30))
        assert t.decode(t.encode(s)) == s


def test_training_compresses_and_is_monotone():
    small = BPETokenizer.train(TEXT, 270)
    big = BPETokenizer.train(TEXT, 400)
    assert big.compression(TEXT) > small.compression(TEXT) > 1.0


def test_incremental_matches_naive():
    text = "abab abab ab ba ba aab " * 5
    merges, _ = train_bpe(text, 270)
    # naive recount reference
    from bpelab.bpe import PRETOKEN_RE, _merge_word
    from collections import Counter
    words = Counter(tuple(w.encode()) for w in PRETOKEN_RE.findall(text))
    ref = []
    for _ in range(len(merges)):
        pc = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pc[p] += c
        pair, cnt = min(pc.items(), key=lambda kv: (-kv[1], kv[0]))
        if cnt < 2:
            break
        nid = 256 + len(ref)
        ref.append(pair)
        nw = Counter()
        for w, c in words.items():
            nw[_merge_word(w, pair, nid)] += c
        words = nw
    assert merges == ref


def test_special_tokens_and_save_load(tmp_path):
    t = BPETokenizer.train(TEXT, 300, special_tokens=["<|endoftext|>"])
    ids = t.encode("hi<|endoftext|>there")
    assert t.special_ids["<|endoftext|>"] in ids
    assert t.decode(ids) == "hi<|endoftext|>there"
    p = tmp_path / "tok.json"
    t.save(p)
    t2 = BPETokenizer.load(p)
    assert t2.encode(TEXT) == t.encode(TEXT)


def test_merges_never_cross_words():
    t = BPETokenizer.train("the cat the cat the cat " * 10, 300)
    for tok in t.vocab.values():
        s = tok.strip(b" ")
        assert b" " not in s


def test_underscore_and_marks_roundtrip():
    t = BPETokenizer.train(TEXT, 300)
    s = "snake_case __init__ e\u0301 \u0300x"
    assert t.decode(t.encode(s)) == s
