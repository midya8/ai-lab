"""CLI: python -m bpekit train FILE -v 400 -o tok.json | encode/decode | eval."""
from __future__ import annotations
import argparse, sys
from .bpe import BPETokenizer, train


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bpekit")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("file"); t.add_argument("-v", "--vocab", type=int, default=400); t.add_argument("-o", "--out", required=True)
    e = sub.add_parser("encode"); e.add_argument("tok"); e.add_argument("text")
    d = sub.add_parser("decode"); d.add_argument("tok"); d.add_argument("ids", type=int, nargs="+")
    v = sub.add_parser("eval", help="train on first 80%% of FILE, report bytes/token on held-out 20%%")
    v.add_argument("file"); v.add_argument("-v", "--vocab", type=int, default=400)
    a = p.parse_args(argv)
    if a.cmd == "train":
        tok = train(open(a.file, encoding="utf-8").read(), a.vocab); tok.save(a.out)
        print(f"saved {len(tok)} tokens to {a.out}")
    elif a.cmd == "encode":
        print(*BPETokenizer.load(a.tok).encode(a.text))
    elif a.cmd == "decode":
        print(BPETokenizer.load(a.tok).decode(a.ids))
    else:
        text = open(a.file, encoding="utf-8").read(); cut = int(len(text) * 0.8)
        tok = train(text[:cut], a.vocab); held = text[cut:]
        print(f"train bytes/token {tok.compression(text[:cut]):.3f}  held-out {tok.compression(held):.3f}  vocab {len(tok)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
