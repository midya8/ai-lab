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
