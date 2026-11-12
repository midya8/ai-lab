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


## BPE-dropout

`tok.encode(text, dropout=0.1, rng=random.Random(0))` skips each applicable merge with probability p (Provilkov et al., 2020), giving varied segmentations of the same text for training-time regularisation; decoding stays lossless and specials are untouched. Default `dropout=0` is the unchanged deterministic path. Only the mechanics are tested here (round-trip, variation, longer sequences as p rises, seeded reproducibility); no downstream model-quality claim is made (5 tests, 20 total).


## Vocab-size sweep

`bpekit.sweep.sweep(text, [sizes])` trains `train_fast` per size on the first 80% and reports train vs held-out bytes/token; `experiments/sweep.py FILE` prints the table. Measured on 775 KB of concatenated stdlib `email` source (single split, one run): vocab 300 → 1.505 held-out, 400 → 1.987, 600 → 2.481, 800 → 2.796, 1200 → 3.249, 1600 → 3.542 bytes/token. Returns diminish steadily; the train/held-out gap peaks near 800 (+0.069) then narrows, and stays small because source code repeats identifiers across the split. Not a claim about natural-language text (3 tests, 23 total).


## Vocabulary utilisation

`bpekit.usage.usage(tok, text)` reports how many merged tokens a text never emits, the unigram entropy of emitted ids, and the most frequent merged tokens. A high unused fraction on held-out text signals a vocab over-fitted to the training split. Mechanics only are tested (3 tests, 26 total); no corpus-level finding is claimed.

## Domain transfer (`bpekit.transfer`, `experiments/transfer.py`)

Train one tokenizer per corpus (vocab 600) and score each on every corpus's held-out tail
(bytes/token, higher is better). Corpora are this monorepo's own READMEs, Python source and JSON.

| trained on \ scored on | prose | python | json |
|---|---|---|---|
| prose | 1.85 | 1.85 | 1.57 |
| python | 1.58 | 2.20 | 1.63 |
| json | 1.54 | 1.75 | 1.90 |

Mean off-domain loss vs the best tokenizer per column: prose 16.6%, python 14.4%, json 18.6%.
Caveat: the diagonal is not always the column maximum here (the prose tokenizer scores as well on
Python as on prose) because held-out tails are small, unrepresentative slices of heterogeneous
files; this is a single run on one corpus split, so read it as indicative, not a benchmark.

## Offsets and chunking

```python
from bpekit import train, encode_with_offsets, chunk_by_tokens
ids, spans = encode_with_offsets(tok, text)      # spans[i] = (start, end) char offsets
chunks = chunk_by_tokens(tok, text, max_tokens=256, overlap=32)
```

Chunks cut only at word boundaries, never exceed `max_tokens` (except a single word longer than the budget), and with `overlap=0` concatenate back to the input. A token covering part of a multi-byte character reports the whole character's span.

## Streaming decode

`StreamDecoder(tok).push(id)` returns the text completed by each token (empty while inside a multi-byte UTF-8 character); `stream_decode(tok, ids)` yields pieces whose concatenation equals `tok.decode(ids)`. A truncated stream flushes as U+FFFD.


## Vocabulary extension (`bpekit.extend`)

`extend(tok, domain_text, n_new)` keeps every existing merge (old token ids unchanged) and continues BPE training on new text, starting from the old tokenizer's segmentation. Special tokens come after the merges, so their ids shift by the number of merges added. Tested: old merges preserved, lossless round-trip, better compression on the domain text without lengthening the original text, deterministic, early stop when no pair repeats (4 tests, 48 total). Model embeddings for the new ids would still need training; no downstream claim is made.

## Fertility (`bpekit.fertility`)

`fertility(tok, text)` encodes each `\w+` word separately and reports mean tokens per word, the 95th-percentile cost, the share of words that are a single token, and the distinct words that cost the most. It is a quick check of how well a vocabulary fits a domain (a code vocabulary on prose, say). It measures segmentation only, not downstream model quality (3 tests).

## Token budgets (`bpekit.truncate`)

`truncate(tok, text, max_tokens, keep='head'|'tail'|'middle')` fits text into a token budget without splitting a UTF-8 character; `middle` keeps head and tail joined by a marker whose token cost is counted. The re-encoded result is guaranteed not to exceed the budget (4 tests, 57 total).

## CLI additions

`python -m bpekit fertility tok.json corpus.txt` prints fertility stats; `python -m bpekit truncate tok.json "text" -n 20 --keep middle` fits text into a token budget (1 test, 58 total).

## Sequence packing

`pack(tok, docs, seq_len)` joins documents with the EOS special and slices the stream into exact `seq_len` rows, returning the rows plus each document's start offset inside its row (useful for attention masks). `drop_last=False` keeps the short final row.
