"""Unigram-LM vs BPE: bytes/token on held-out stdlib text at equal vocab size.
Run: PYTHONPATH=src python experiments/unigram_vs_bpe.py
"""
import glob, os, sysconfig
from bpelab import BPETokenizer, UnigramTokenizer

def load(fs):
    return "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in fs)

lib = sysconfig.get_paths()["stdlib"]
files = sorted(glob.glob(os.path.join(lib, "*.py")))
train, test = load(files[0::2][:20])[:60_000], load(files[1::2][:10])[:40_000]
print("vocab  bpe_heldout  unigram_heldout")
for v in (384, 512, 768):
    b = BPETokenizer.train(train, v)
    u = UnigramTokenizer.train(train, v, seed_size=3000)
    print(f"{v:5d}  {b.compression(test):11.3f}  {u.compression(test):15.3f}")
