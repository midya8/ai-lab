from bpelab import BPETokenizer, roundtrip_failures, worst_words

TEXT = "the cat sat on the mat. the dog ate the antidisestablishmentarianism " * 20


def test_roundtrip_ok_on_unicode():
    tok = BPETokenizer.train(TEXT, 300)
    assert roundtrip_failures(tok, ["héllo wörld", "日本語", "", "a\nb\t c"]) == []


def test_roundtrip_detects_lossy_tokenizer():
    class Lossy:
        def encode(self, t):
            return list(t.lower().encode())

        def decode(self, ids):
            return bytes(ids).decode()

    assert roundtrip_failures(Lossy(), ["abc", "ABC", "Xy"], limit=1) == ["ABC"]


def test_worst_words_orders_by_tokens():
    tok = BPETokenizer.train("a b c d e f " * 10 + "the cat " * 50, 270)
    ww = worst_words(tok, "the cat zqxjvk the", top=2)
    assert ww[0][0] == "zqxjvk" and ww[0][1] >= ww[1][1]
    assert len(ww) == 2
