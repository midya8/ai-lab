from bpelab import BPETokenizer

TEXT = "the cat sat on the mat the cat ate the rat " * 20


def test_explain_matches_encode():
    tok = BPETokenizer.train(TEXT, 290)
    for w in ["the", "cat", "ratty", "zzz"]:
        steps = tok.explain(w)
        final = steps[-1][3] if steps else [bytes([b]) for b in w.encode()]
        assert b"".join(final) == w.encode()
        assert final == [tok.vocab[i] for i in tok._encode_word(w.encode())]


def test_explain_ranks_increase_pieces_shrink():
    tok = BPETokenizer.train(TEXT, 290)
    steps = tok.explain("the")
    assert steps and all(len(steps[i + 1][3]) < len(steps[i][3]) for i in range(len(steps) - 1))
    assert tok.explain("a") == []
