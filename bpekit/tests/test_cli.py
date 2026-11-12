import io, contextlib
from bpekit import train
from bpekit.__main__ import main

TEXT = "the quick brown fox jumps over the lazy dog. " * 40


def run(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(args) == 0
    return buf.getvalue()


def test_fertility_and_truncate_cli(tmp_path):
    tok_path, corpus = tmp_path / "t.json", tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    train(TEXT, 300).save(str(tok_path))
    out = run(["fertility", str(tok_path), str(corpus), "--worst", "2"])
    assert out.startswith("words 360") and out.count("\n") == 3
    from bpekit import BPETokenizer
    tok = BPETokenizer.load(str(tok_path))
    for keep in ("head", "tail", "middle"):
        res = run(["truncate", str(tok_path), TEXT, "-n", "20", "--keep", keep]).rstrip("\n")
        assert 0 < len(tok.encode(res)) <= 20
