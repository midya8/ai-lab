# bpelab

Byte-level BPE tokenizer trainer in pure Python (GPT-2 style): 256-byte base vocabulary, so every string round-trips losslessly; regex pre-tokenisation so merges never cross words; incremental pair-count updates during training; special tokens; JSON save/load.

```python
from bpelab import BPETokenizer
tok = BPETokenizer.train(open("corpus.txt").read(), vocab_size=512, special_tokens=["<|endoftext|>"])
ids = tok.encode("hello world"); assert tok.decode(ids) == "hello world"
print(tok.compression(text))   # bytes per token
```

Tests: `PYTHONPATH=src python -m pytest -q` (7 tests, including a check that the incremental trainer yields the same merges as a naive full recount, and round-trips on random unicode).

Limits: training is O(merges x affected words) in pure Python, fine for MB-scale corpora, not GB. Tie-breaking is deterministic (smallest pair), which differs from other libraries, so merges are not interchangeable with GPT-2's.

## Compression vs vocab size (`experiments/curve.py`)

Trained on 200k chars of stdlib source, evaluated on 100k chars from disjoint files (bytes per token):

| vocab | train | held-out |
|---|---|---|
| 256 | 1.000 | 1.000 |
| 384 | 2.109 | 1.791 |
| 512 | 2.529 | 2.016 |
| 1024 | 3.293 | 2.504 |

Held-out gains show diminishing returns and a widening train/held-out gap (memorised file-specific tokens) as vocab grows; this is a small-corpus result, not a claim about large-scale tokenizers.

## Unigram-LM tokenizer (v0.2)

`UnigramTokenizer` (Kudo 2018): seed vocab of bytes + frequent substrings, EM via forward-backward over each word's segmentation lattice, pruning to the vocab budget, Viterbi encoding, byte fallback (lossless). 7 tests incl. EM likelihood monotonicity and Viterbi optimality.

Held-out bytes/token vs BPE (`experiments/unigram_vs_bpe.py`, small 60k-char train, single run):

| vocab | BPE | Unigram |
|---|---|---|
| 384 | 1.871 | 1.857 |
| 512 | 2.121 | 2.055 |
| 768 | 2.373 | 2.484 |

Mixed: unigram is slightly worse at small vocabs and better at 768. Pruning uses a cheap proxy (expected count x -logp) rather than SentencePiece's exact re-segmentation loss, and the corpus is tiny, so don't read this as a general ranking.

## Unigram sampling (subword regularisation)

`UnigramTokenizer.encode(text, sample_alpha=0.5, rng=random.Random(0))` draws a random segmentation by forward-filtering backward-sampling, with probability proportional to p(seg)^alpha. alpha=1 is the LM posterior; large alpha approaches Viterbi. A test verifies the empirical frequencies against the exact posterior on a toy vocabulary.

## N-best segmentations

`UnigramTokenizer.nbest("word", n=3)` returns the top-n segmentations with log-probs via an exact k-best lattice DP (first entry equals Viterbi). Tests compare against brute-force enumeration.

## Tokenizer statistics

`bpelab.token_stats(tok, text)` works with any tokenizer exposing `encode` and `vocab_size` and returns fertility (tokens per word), bytes per token, vocabulary utilisation and Renyi (alpha=2.5) efficiency. 4 tests.

## Dead-vocabulary check

`bpelab.dead_vocab(tok, held_out_text)` lists token ids that never occur in the given text, the dead fraction, and the most frequent ids. Useful for spotting a vocab that is over-sized for its domain. 3 tests.

## Comparing tokenizers

`bpelab.compare_tokenizers({"bpe-300": bpe, "unigram": uni}, text)` runs `token_stats` for each named tokenizer on the same text and returns rows sorted by fertility. 2 tests.
