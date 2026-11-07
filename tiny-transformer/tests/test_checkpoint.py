import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import (BPETokenizer, CharTokenizer, GPT, GPTConfig, evaluate,  # noqa: E402
                     generate_cached, load_checkpoint, save_checkpoint)
from tinygpt.cli import main  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog. " * 20


def _model(tok):
    return GPT(GPTConfig(tok.vocab_size, 16, 1, 2, 16), seed=1)


@pytest.mark.parametrize("kind", ["char", "bpe"])
def test_roundtrip_is_exact(tmp_path, kind):
    tok = CharTokenizer(TEXT) if kind == "char" else BPETokenizer.train(TEXT, 280)
    m = _model(tok)
    p = str(tmp_path / "c.npz")
    save_checkpoint(p, m, tok)
    m2, tok2 = load_checkpoint(p)
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    assert tok2.decode(tok.encode("fox")) == "fox"
    ids = tok.encode("the q")
    assert list(generate_cached(m, ids, 10, seed=3)) == list(generate_cached(m2, ids, 10, seed=3))


def test_untrained_perplexity_near_vocab():
    tok = CharTokenizer(TEXT)
    r = evaluate(_model(tok), np.array(tok.encode(TEXT)))
    assert abs(r["perplexity"] - tok.vocab_size) / tok.vocab_size < 0.25


def test_evaluate_deterministic_and_batch_invariant():
    tok = CharTokenizer(TEXT)
    m, d = _model(tok), np.array(tok.encode(TEXT))
    a, b = evaluate(m, d, batch=1), evaluate(m, d, batch=7)
    assert abs(a["loss"] - b["loss"]) < 1e-5 and a["tokens"] == b["tokens"]


def test_evaluate_too_short():
    tok = CharTokenizer(TEXT)
    with pytest.raises(ValueError):
        evaluate(_model(tok), np.array([0, 1, 2]))


def test_cli_generate_and_eval(tmp_path, capsys):
    tok = CharTokenizer(TEXT)
    p = str(tmp_path / "c.npz")
    save_checkpoint(p, _model(tok), tok)
    f = tmp_path / "d.txt"
    f.write_text(TEXT)
    main(["generate", p, "--prompt", "the", "-n", "5"])
    assert capsys.readouterr().out.startswith("the")
    main(["eval", p, "--data", str(f)])
    assert "perplexity" in capsys.readouterr().out
