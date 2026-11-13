"""Compression-vs-vocab-size curve on held-out text.

Train on one set of stdlib .py files, evaluate bytes/token on disjoint files.
Run: PYTHONPATH=src python experiments/curve.py
"""
import glob, os, sys, sysconfig
from bpelab import BPETokenizer

def load(files):
    return "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)

def main(sizes=(256, 288, 384, 512, 768, 1024)):
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(lib, "*.py")))
    train_files, test_files = files[0::2][:40], files[1::2][:20]
    train, test = load(train_files)[:200_000], load(test_files)[:100_000]
    print(f"train {len(train)} chars, held-out {len(test)} chars")
    print("vocab  train_B/tok  heldout_B/tok")
    rows = []
    for v in sizes:
        tok = BPETokenizer.train(train, vocab_size=v)
        a, b = tok.compression(train), tok.compression(test)
        rows.append((v, a, b)); print(f"{v:5d}  {a:11.3f}  {b:13.3f}")
    return rows

if __name__ == "__main__":
    main()
