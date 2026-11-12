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
    f = sub.add_parser("fertility", help="tokens-per-word stats of FILE under a saved tokenizer")
    f.add_argument("tok"); f.add_argument("file"); f.add_argument("--worst", type=int, default=5)
    c = sub.add_parser("truncate", help="fit TEXT into a token budget")
    c.add_argument("tok"); c.add_argument("text"); c.add_argument("-n", "--max-tokens", type=int, required=True)
    c.add_argument("--keep", choices=["head", "tail", "middle"], default="head")
    a = p.parse_args(argv)
    if a.cmd == "train":
        tok = train(open(a.file, encoding="utf-8").read(), a.vocab); tok.save(a.out)
        print(f"saved {len(tok)} tokens to {a.out}")
    elif a.cmd == "encode":
        print(*BPETokenizer.load(a.tok).encode(a.text))
    elif a.cmd == "decode":
        print(BPETokenizer.load(a.tok).decode(a.ids))
    elif a.cmd == "fertility":
        from .fertility import fertility
        r = fertility(BPETokenizer.load(a.tok), open(a.file, encoding="utf-8").read(), a.worst)
        print(f"words {r['words']}  mean {r['mean']:.3f}  p95 {r['p95']}  single {r['single_frac']:.3f}")
        for w, n in r["worst"]:
            print(f"  {n:3d}  {w}")
    elif a.cmd == "truncate":
        from .budget import truncate
        print(truncate(BPETokenizer.load(a.tok), a.text, a.max_tokens, a.keep))
    else:
        text = open(a.file, encoding="utf-8").read(); cut = int(len(text) * 0.8)
        tok = train(text[:cut], a.vocab); held = text[cut:]
        print(f"train bytes/token {tok.compression(text[:cut]):.3f}  held-out {tok.compression(held):.3f}  vocab {len(tok)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
