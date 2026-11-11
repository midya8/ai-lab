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
