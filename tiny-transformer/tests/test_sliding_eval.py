import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import CharTokenizer, GPT, GPTConfig, evaluate, evaluate_sliding, save_checkpoint  # noqa: E402
from tinygpt.cli import main  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog. " * 20


def _setup():
    tok = CharTokenizer(TEXT)
    m = GPT(GPTConfig(tok.vocab_size, 16, 1, 2, 16), seed=1)
    return m, tok, np.array(tok.encode(TEXT))


def test_full_stride_matches_plain_eval_on_aligned_data():
    m, _, data = _setup()
    data = data[:16 * 10 + 1]
    a = evaluate_sliding(m, data, stride=16)
    b = evaluate(m, data)
    assert a["tokens"] == b["tokens"] == 160
    assert a["loss"] == pytest.approx(b["loss"], rel=1e-6)


def test_every_target_token_counted_once():
    m, _, data = _setup()
    for stride in (1, 3, 5, 16):
        assert evaluate_sliding(m, data, stride=stride)["tokens"] == len(data) - 1


def test_stride_one_matches_manual_full_context_scoring():
    m, _, data = _setup()
    data = data[:40]
    T = 16
    total = 0.0
    for t in range(len(data) - 1):  # score token t+1 with up to T tokens of context
        s = max(0, t + 1 - T)
        x = data[s:t + 1]
        logits, _, _ = m.forward(np.pad(x, (0, T - len(x)))[None])
        row = logits[0, len(x) - 1]
        z = row - row.max()
        total += -(z - np.log(np.exp(z).sum()))[data[t + 1]]
    r = evaluate_sliding(m, data, stride=1)
    assert r["tokens"] == len(data) - 1
    assert r["loss"] == pytest.approx(total / (len(data) - 1), rel=1e-6)


def test_short_data_and_bad_stride():
    m, _, data = _setup()
    r = evaluate_sliding(m, data[:8], stride=4)
    assert r["tokens"] == 7
    with pytest.raises(ValueError):
        evaluate_sliding(m, data, stride=0)
    with pytest.raises(ValueError):
        evaluate_sliding(m, data, stride=17)


def test_cli_stride(tmp_path, capsys):
    m, tok, _ = _setup()
    ck = str(tmp_path / "c.npz"); save_checkpoint(ck, m, tok)
    d = tmp_path / "d.txt"; d.write_text(TEXT)
    main(["eval", ck, "--data", str(d), "--stride", "4"])
    assert "sliding(stride=4)" in capsys.readouterr().out
