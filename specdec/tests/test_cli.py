import json
import pytest
from specdec.__main__ import run, main

TEXT = (b"the quick brown fox jumps over the lazy dog. " * 40)


def test_run_fields_and_sanity():
    r = run(TEXT, n=600)
    assert 0 < r["alpha_calibrated"] <= 1 and r["k"] >= 1
    assert r["tokens_per_target_call"] >= 1.0
    assert abs(r["alpha_calibrated"] - r["alpha_measured"]) < 0.25


def test_deterministic():
    assert run(TEXT, n=300, seed=3) == run(TEXT, n=300, seed=3)


def test_too_short():
    with pytest.raises(ValueError):
        run(b"abc")


def test_main(tmp_path, capsys):
    p = tmp_path / "a.txt"; p.write_bytes(TEXT)
    assert main([str(p), "--n", "300"]) == 0
    assert "k" in json.loads(capsys.readouterr().out)
    q = tmp_path / "b.txt"; q.write_bytes(b"x")
    assert main([str(q)]) == 2
