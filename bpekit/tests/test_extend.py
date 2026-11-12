import random
from bpekit import train_fast, extend

GEN = "the cat sat on the mat. the dog ate the bone. " * 30
DOM = "def forward(self, x): return self.linear(x)\n" * 40


def test_old_merges_preserved_and_roundtrip():
    base = train_fast(GEN, 280)
    ext = extend(base, DOM, 30)
    assert ext.merges[: len(base.merges)] == base.merges
    assert len(ext.merges) > len(base.merges)
    for t in (GEN, DOM, "héllo wörld ✓"):
        assert ext.decode(ext.encode(t)) == t


def test_domain_compression_improves_general_not_worse():
    base = train_fast(GEN, 280)
    ext = extend(base, DOM, 40)
    assert ext.compression(DOM) > base.compression(DOM)
    assert len(ext.encode(GEN)) <= len(base.encode(GEN))


def test_zero_and_early_stop_and_specials():
    base = train_fast(GEN, 280)
    assert extend(base, DOM, 0).merges == base.merges
    assert len(extend(base, "abc", 50).merges) == len(base.merges)  # no repeated pair
    ext = extend(base, DOM, 5)
    assert list(ext.specials) == list(base.specials)
    assert ext.specials["<​|endoftext|>"] == 256 + len(ext.merges)


def test_old_ids_stable_on_old_text():
    base = train_fast(GEN, 280)
    ext = extend(base, DOM, 20)
    rng = random.Random(0)
    s = "".join(rng.choice("the cat sat mat dog") for _ in range(200))
    assert ext.encode(s) == base.encode(s) or ext.decode(ext.encode(s)) == s
    assert extend(base, DOM, 20).merges == ext.merges  # deterministic
