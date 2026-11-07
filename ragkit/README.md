# ragkit

Local retrieval-augmented generation over your own documents. No API
keys, no server, no network on the default path. Python standard
library only — no numpy, no scikit-learn, no dependencies at all.

Built in November 2026 as a portfolio project to show how a RAG
system actually works under the hood. It is not client work, and the
example documents describe a fictional business, not a real one.

## What it does

- **Ingest** — reads Markdown and plain-text files from a directory,
  splits them into overlapping, sentence-aware chunks, and records
  each chunk's source file, line range, and heading context.
- **Retrieve** — TF-IDF and BM25 scoring implemented from scratch in
  the stdlib, with light stemming, a stopword list, and an
  exact-phrase boost. A `Retriever` class is usable as a library.
- **Answer** — a pluggable `Answerer` protocol. The default
  extractive answerer quotes the most relevant chunks with citations
  like `[returns.md:8-22]`, so every claim traces back to a source
  line range. An optional OpenAI-compatible answerer implements the
  same protocol and reads its key from the environment only.
- **Evaluate** — an `eval` command runs a golden set and reports
  recall@k, citation precision, and which questions failed, as text
  and JSON.

## Quickstart

```bash
python -m ragkit index examples/docs /tmp/ragkit-index.json
python -m ragkit ask /tmp/ragkit-index.json "How long do I have to return a bike?"
python -m ragkit eval /tmp/ragkit-index.json eval/golden.json
```

Example answer:

```
Based on the indexed documents:

[returns.md:8-22] Returns are accepted within 30 days of purchase. The item
must be unused and in its original packaging. Bring your receipt or order
confirmation with you, since returns cannot be processed without proof of
purchase. Refunds go back to the original payment method within 5 business
days. Exchanges are free and can be done in store on the same visit.
```

Exit codes: `0` ok, `1` no results / eval below `--fail-below`,
`2` invalid index, query, or arguments.

## Library use

```python
from ragkit import load_documents, Retriever, ExtractiveAnswerer

chunks = load_documents("examples/docs")
retriever = Retriever(chunks, scoring="bm25")
scored = retriever.search("What does a standard tune-up cost?", k=5)
answer = ExtractiveAnswerer().answer(
    "What does a standard tune-up cost?",
    [s.chunk for s in scored],
)
print(answer.text)  # quoted excerpts with [file:start-end] citations
```

## Architecture

```
ragkit/
  chunk.py    sentence-aware chunking, line ranges, heading context
  ingest.py   read a docs directory, save/load the JSON index
  retrieve.py TF-IDF + BM25 from scratch, phrase boost, Retriever
  answer.py   Answerer protocol, extractive + OpenAI-compatible answerers
  eval.py     golden-set evaluation: recall@k, citation precision
  __main__.py CLI: index / ask / eval
tests/        unittest suite (see below)
examples/docs 6 demo Markdown files (fictional bike shop, labelled demo data)
eval/golden.json  8 questions with expected sources and answer spans
```

Three retrieval details worth knowing:

1. **Light stemming + stopwords.** Token matching stems word
   variants (`return` ~ `returns`, `close` ~ `closed`) and drops
   function words, so a query matches on content terms instead of
   on "a" or "the". The stemmer is deliberately small (documented
   in `retrieve.py`); it is not a Porter stemmer.
2. **Relative-score cutoff.** Top-k retrieval always returns k
   chunks, but the tail is often barely relevant. Answers are built
   only from chunks scoring at least half the top score.
3. **Two-term rule for quoted chunks.** Beyond the top chunk, a
   chunk is only quoted if it shares at least two distinct content
   terms with the query. A chunk matching on a single word ("bike"
   in a bike-shop corpus) is on-topic but rarely supports the
   answer, and citing it sinks citation precision. Recall is still
   measured on the full top-k list.

The optional LLM answerer (`OpenAICompatibleAnswerer`) takes its key
from `RAGKIT_API_KEY` (or `OPENAI_API_KEY`) and its endpoint from
`OPENAI_BASE_URL`. It is never called by the tests or the default
path.

## Measured results

Run on 2026-11-07, Python 3.12, against the shipped demo docs
(6 files, 6 chunks) and `eval/golden.json` (8 questions):

| scorer | mean recall@5 | mean citation precision | passed |
| ------ | ------------- | ----------------------- | ------ |
| BM25 (default) | 1.000 | 1.000 | 8/8 |
| TF-IDF | 1.000 | 1.000 | 8/8 |

Tests:

```
$ python -m unittest discover -s tests
Ran 59 tests in 2.1s

OK
```

The suite covers chunking overlap, sentence-boundary behaviour,
line ranges and heading context, TF-IDF/BM25 ranking (a known query
returns the known document first), the exact-phrase boost, citation
formats, eval metric math, empty-corpus handling, CLI exit codes,
and offline determinism (the same question twice gives the same
answer). CI runs the suite and the CLI smoke test on Python
3.10–3.13 (`.github/workflows/ci.yml`).

## Limitations

- **Keyword retrieval, not semantic search.** There are no
  embeddings, so a query using different words than the documents
  ("refund window" vs "returns") may miss. The stemmer and phrase
  boost narrow this gap; they do not close it.
- **Light stemmer.** Word variants that differ by more than a
  suffix (`expire` vs `expires`) do not conflate, and aggressive
  suffix-stripping can occasionally conflate unrelated words.
- **Extractive answers only, by default.** The offline answerer
  quotes; it does not synthesise, summarise across documents, or
  say "I don't know" beyond reporting no relevant documents.
  Anything needing real generation needs the optional LLM answerer
  (and an API key).
- **No server, no incremental indexing.** There is intentionally
  no `serve` command; re-indexing rebuilds the whole index, which
  is fine for personal document sets and not designed for large
  corpora.
- **Small-corpus effects.** IDF and the two-term rule behave
  differently on very small document sets; the golden set above is
  small (8 questions, 6 documents), so treat the 1.000 scores as
  "works on its demo" rather than a benchmark.
