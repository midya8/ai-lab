# ragkit

Local RAG over your own documents, using only the standard library. I wanted to see how few moving parts a working version needs: no embeddings, no server, no network on the default path.

```bash
cd ragkit
python -m ragkit index examples/docs /tmp/ragkit-index.json
python -m ragkit ask /tmp/ragkit-index.json "How long do I have to return a bike?"
python -m ragkit eval /tmp/ragkit-index.json eval/golden.json
python -m unittest discover -s tests
```

```
Based on the indexed documents:

[returns.md:8-22] Returns are accepted within 30 days of purchase. The item
must be unused and in its original packaging. ...
```

It reads Markdown and text files into overlapping, sentence-aware chunks that keep their file, line range and heading. Retrieval is TF-IDF or BM25 written from scratch, with a small stemmer, stopwords and an exact-phrase boost. The default answerer quotes the best chunks with citations, so every claim points back to specific lines. There's an optional OpenAI-compatible answerer if you want generated text.

Two small rules made the citations much cleaner. Only chunks scoring at least half the top score get used. And beyond the top chunk, a chunk has to share at least two content words with the question, because in a bike-shop corpus everything matches "bike".

On the 8-question golden set over the 6 demo docs (a fictional bike shop), both BM25 and TF-IDF get recall@5 1.000 and citation precision 1.000. That's "works on its demo", not a benchmark.

It's keyword retrieval, so "refund window" won't find "returns", and the stemmer is deliberately small. [`docqa`](../docqa) builds on this with abstention and spelling correction.
