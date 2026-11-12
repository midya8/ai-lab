import math
from bpekit import train, zipf_fit

TEXT = "the quick brown fox jumps over the lazy dog. " * 20 + "a b c d e f g h"


def test_basic_fields():
    t = train(TEXT, 300)
    r = zipf_fit(t, TEXT)
    assert r["tokens"] == len(t.encode(TEXT))
    assert r["slope"] < 0 and 0 <= r["r2"] <= 1


def test_top_restricts_and_degenerate():
    t = train(TEXT, 300)
    assert zipf_fit(t, TEXT, top=3)["distinct"] == zipf_fit(t, TEXT)["distinct"]
    r = zipf_fit(t, "a")
    assert r["slope"] is None and r["r2"] is None


def test_exact_power_law_math():
    class Fake:
        def encode(self, s):
            ids = []
            for rank in range(1, 6):
                ids += [rank] * (600 // rank)
            return ids
    r = zipf_fit(Fake(), "")
    assert abs(r["slope"] + 1) < 0.02 and r["r2"] > 0.999
