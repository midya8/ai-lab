import itertools
import math
from bpelab import UnigramTokenizer
from bpelab.unigram import nbest, viterbi


def _all_segs(w, logp):
    out = []
    for cuts in itertools.product([0, 1], repeat=len(w) - 1):
        segs, st = [], 0
        for i, c in enumerate(cuts, 1):
            if c:
                segs.append(w[st:i]); st = i
        segs.append(w[st:])
        if all(x in logp for x in segs):
            out.append((sum(logp[x] for x in segs), segs))
    return sorted(out, key=lambda t: -t[0])


LOGP = {b"a": -2.0, b"b": -2.5, b"c": -3.0, b"ab": -3.0, b"bc": -3.5, b"abc": -5.0, b"abca": -6.0}


def test_nbest_matches_bruteforce():
    w = b"abca"
    got = nbest(w, LOGP, 4, n=4)
    want = _all_segs(w, LOGP)[:4]
    assert [round(s, 9) for _, s in got] == [round(s, 9) for s, _ in want]


def test_first_is_viterbi_and_sorted():
    got = nbest(b"abc", LOGP, 4, n=5)
    assert got[0][0] == viterbi(b"abc", LOGP, 4)[0]
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True)
    assert len(got) == len(_all_segs(b"abc", LOGP))


def test_tokenizer_nbest_decodes_to_word():
    t = UnigramTokenizer.train("the cat sat on the mat. " * 40, 300, seed_size=300)
    res = t.nbest("the cat", 3)
    assert len(res) == 3
    for ids, sc in res:
        assert t.decode(ids) == "the cat" and math.isfinite(sc)
