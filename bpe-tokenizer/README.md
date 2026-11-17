# bpe-tokenizer

A byte-level BPE tokenizer trainer in plain Python, GPT-2 style: 256 byte base vocab so everything round-trips, regex pre-tokenisation so merges don't cross words, special tokens, JSON save/load. Later I added a unigram-LM tokenizer (Kudo 2018) to compare against it.

```python
from bpe_tokenizer import BPETokenizer

tok = BPETokenizer.train(open("corpus.txt").read(), vocab_size=512,
                         special_tokens=["<|endoftext|>"])
ids = tok.encode("hello world")
assert tok.decode(ids) == "hello world"
```

Run the tests with `cd bpe-tokenizer && PYTHONPATH=src python -m pytest -q`. One of them checks that the incremental pair-count trainer produces exactly the same merges as a naive full recount.

## Compression vs vocab size

Trained on 200k chars of Python stdlib source, measured on 100k chars from other files (bytes per token, `experiments/curve.py`):

| vocab | train | held-out |
|---|---|---|
| 256 | 1.000 | 1.000 |
| 384 | 2.109 | 1.791 |
| 512 | 2.529 | 2.016 |
| 1024 | 3.293 | 2.504 |

Diminishing returns, and the train/held-out gap keeps widening as the vocab starts memorising file-specific strings.

BPE vs unigram on a smaller 60k-char corpus was a wash: unigram is slightly worse at 384 and 512 and better at 768 (2.484 vs 2.373). Tiny corpus, one run.

## Other bits

- Unigram tokenizer with EM over the segmentation lattice, Viterbi encoding, n-best segmentations and subword-regularisation sampling.
- `token_stats` (fertility, bytes/token, Renyi efficiency), `dead_vocab`, `segmentation_agreement` between two tokenizers, and `vocab_size_sweep` to find where extra merges stop helping.

Training is pure Python, so fine for MB-sized corpora, not GB. Ties break on the smallest pair, so merges won't match GPT-2's exactly.
