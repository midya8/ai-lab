# bpekit

Byte-level BPE tokenizer trainer, encoder and decoder written from scratch (GPT-2 style pre-splitting, no tokenizer libraries).

- Trains deterministic merges (ties broken by pair id); stops when no pair occurs twice.
- Lossless for any Unicode string (falls back to raw bytes); atomic special tokens.
- Save/load as JSON; `compression()` reports bytes per token.

```python
from bpekit import train
tok = train(open("corpus.txt").read(), vocab_size=1000)
ids = tok.encode("hello world"); assert tok.decode(ids) == "hello world"
```

Test: `PYTHONPATH=src python -m pytest -q` (7 tests). Training is pure Python and O(merges x unique words); fine for MB-scale corpora, not GB.


## CLI

```bash
PYTHONPATH=src python -m bpekit train corpus.txt -v 400 -o tok.json
PYTHONPATH=src python -m bpekit encode tok.json "hello world"
PYTHONPATH=src python -m bpekit eval corpus.txt -v 400   # held-out bytes/token
```


## Fast trainer

`train_fast` produces identical merges to `train` (tested on random corpora across vocab sizes) but updates pair counts incrementally via a pair-to-words index. Measured here on 4.4 MB of stdlib Python source, vocab 1500 (1244 merges learned before no pair repeated): `train` 89.5 s vs `train_fast` 5.8 s (~15x), same merges and 2.976 bytes/token. Single run on one machine; timing is indicative only.


## GPT-2 / HuggingFace-style export

`export_gpt2(tok, "vocab.json", "merges.txt")` writes the merge table in the GPT-2 byte-to-unicode alphabet; `import_gpt2(...)` rebuilds the tokenizer and rejects files whose ids disagree with merge order (4 tests, 15 total). Only the merge table is portable: this package's pre-split regex is simpler than GPT-2's, so other libraries can tokenise contractions/punctuation differently. Not verified against an external library here.
