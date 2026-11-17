import numpy as np

from hopfield import studies as S
from hopfield.glyphs import DIGITS, digit_patterns, render
from hopfield.__main__ import main


def test_digits_shapes_distinct_pm1():
    X = digit_patterns()
    assert X.shape == (10, 35) and set(np.unique(X)) == {-1.0, 1.0}
    assert len({tuple(x) for x in X}) == 10
    assert all(d.shape == (7, 5) for d in DIGITS.values())


def test_digits_are_correlated_not_random():
    X = digit_patterns()
    G = np.abs(X @ X.T / 35 - np.eye(10))
    assert G.max() > 0.4          # some pairs share a lot of pixels


def test_render_roundtrip_lines():
    s = render(digit_patterns([8])[0])
    assert len(s.split("\n")) == 7 and all(len(r) == 5 for r in s.split("\n"))


def test_glyph_study_modern_beats_hebbian():
    rows = {r["rule"]: r for r in S.glyph_study(trials=5)}
    assert rows["dam-exp"]["stored_stable"] == 1.0
    assert rows["dam-exp"]["flip10"] > rows["hebbian"]["flip10"] + 0.5
    assert rows["pseudoinverse"]["stored_stable"] == 1.0


def test_capacity_study_small_shows_hebbian_cliff():
    rows = S.capacity_study(n=60, ps=(3, 30), seeds=1, queries=10, rules=("hebbian", "dam-exp"))
    assert rows[0]["hebbian"] > 0.8 and rows[1]["hebbian"] < 0.2
    assert rows[1]["dam-exp"] == 1.0


def test_beta_study_has_sharp_transition():
    rows = S.beta_study(betas=(1.0, 30.0), seeds=1)
    assert rows[0]["retrieval"] == 0.0 and rows[0]["effective_patterns"] > 20
    assert rows[1]["retrieval"] == 1.0 and rows[1]["effective_patterns"] < 1.1


def test_cli(capsys):
    assert main(["glyphs"]) == 0
    assert "dam-exp" in capsys.readouterr().out
    assert main([]) == 2
