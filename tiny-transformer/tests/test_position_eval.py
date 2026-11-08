import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import CharTokenizer, GPT, GPTConfig, evaluate, evaluate_by_position, save_checkpoint  # noqa: E402
from tinygpt.cli import main  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog. " * 20


def _setup():
    tok = CharTokenizer(TEXT)
    m = GPT(GPTConfig(tok.vocab_size, 16, 1, 2, 16), seed=1)
    return m, tok, np.array(tok.encode(TEXT))


def test_per_position_mean_matches_overall_loss():
    m, _, data = _setup()
    r = evaluate_by_position(m, data, bins=4)
    assert len(r["per_position"]) == 16
    assert np.mean(r["per_position"]) == pytest.approx(evaluate(m, data)["loss"], rel=1e-6)


def test_bins_cover_window_and_average_correctly():
    m, _, data = _setup()
    r = evaluate_by_position(m, data, bins=4)
    assert [(b["start"], b["end"]) for b in r["bins"]] == [(0, 4), (4, 8), (8, 12), (12, 16)]
    per = np.array(r["per_position"])
    assert r["bins"][1]["loss"] == pytest.approx(per[4:8].mean())
    assert r["gain"] == pytest.approx(r["bins"][0]["loss"] - r["bins"][-1]["loss"])


def test_invalid_args():
    m, _, data = _setup()
    with pytest.raises(ValueError):
        evaluate_by_position(m, data, bins=0)
    with pytest.raises(ValueError):
        evaluate_by_position(m, data[:5])


def test_cli_by_position(tmp_path, capsys):
    m, tok, _ = _setup()
    ck, d = str(tmp_path / "c.npz"), tmp_path / "d.txt"
    save_checkpoint(ck, m, tok)
    d.write_text(TEXT)
    main(["eval", ck, "--data", str(d), "--by-position", "2"])
    out = capsys.readouterr().out
    assert "pos    0-8" in out and "context gain" in out
