# bpekit

My more practical BPE tokenizer: train, encode, decode, plus the boring utilities you actually need around a tokenizer (offsets, chunking, truncation, packing, streaming decode). GPT-2 style pre-splitting and lossless on any Unicode string, with no tokenizer libraries. [`bpe-tokenizer`](../bpe-tokenizer) is the earlier, more experimental one that compares BPE with a unigram LM.

```python
from bpekit import train, encode_with_offsets, chunk_by_tokens

tok = train(open("corpus.txt").read(), vocab_size=1000)
ids = tok.encode("hello world"); assert tok.decode(ids) == "hello world"
ids, spans = encode_with_offsets(tok, "hello world")
chunks = chunk_by_tokens(tok, open("corpus.txt").read(), max_tokens=256, overlap=32)
```

```bash
cd bpekit
PYTHONPATH=src python -m bpekit train corpus.txt -v 400 -o tok.json
PYTHONPATH=src python -m bpekit encode tok.json "hello world"
PYTHONPATH=src python -m bpekit eval corpus.txt -v 400
PYTHONPATH=src python -m pytest -q
```

## A few numbers

- `train_fast` updates pair counts incrementally and gives exactly the same merges as the naive trainer. On 4.4 MB of stdlib source at vocab 1500 it took 5.8 s vs 89.5 s.
- Vocab sweep on 775 KB of stdlib `email` source (held-out bytes per token): 300 gives 1.51, 800 gives 2.80, 1600 gives 3.54. Steady diminishing returns.
- Tokenizers trained on prose, Python and JSON lose 14-19% compression when used off-domain. That's a single small split, so don't lean on it.

## Other bits

GPT-2 style vocab/merges export, BPE-dropout, extending a vocab without changing existing ids, fertility and coverage stats, a finder for under-trained tokens, round-trip fuzzing, and a small constraint helper for classification-by-generation.

The trainer is pure Python, so fine for MB-scale corpora, not GB. The pre-split regex is simpler than GPT-2's, so only the merge table is portable.
