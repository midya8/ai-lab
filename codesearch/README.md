# codesearch

Code search for Python repos. Ask in English or paste an identifier, and you get the function back instead of a pile of grep lines. NumPy only.

```bash
cd codesearch
PYTHONPATH=src python -m codesearch search --root .. "write json to a temp file then rename it" --show
PYTHONPATH=src python -m codesearch search --root .. '`parse_diff`'        # exact symbol gets pinned first
PYTHONPATH=src python -m codesearch similar --root .. llm-router/llmrouter/breaker.py::CircuitBreaker
PYTHONPATH=src python -m codesearch dups --root ..                         # near-duplicate functions
PYTHONPATH=src python -m codesearch hotspots --root .. --min 5             # complexity x references
PYTHONPATH=src python -m codesearch eval bench/queries_dev.json
PYTHONPATH=src python -m pytest -q
```

Every function, method and class becomes its own chunk (via the AST) with its signature, docstring, identifiers and comments. Identifiers are split so `parseHTTPResponse`, `parse_http_response` and "parse an HTTP response" all match. Ranking is BM25F over weighted fields (name, owner class, docstring, signature, body, path), and tests are demoted unless you ask for them. The index is incremental (SHA-256 per file), and there's a `watch` mode and a JSON-lines `serve` mode.

## How well it works

Two sets of 26 natural-language queries over the other projects in this repo:

| mode | dev MRR | held-out MRR |
|---|---|---|
| name only (grep-like) | 0.244 | - |
| flat BM25 | 0.522 | 0.520 |
| BM25F | 0.694 | 0.578 |
| LSA | 0.414 | 0.402 |
| hybrid | 0.602 | 0.542 |

BM25F is the default. LSA and the hybrid didn't beat it at this size, so LSA is only used for `similar` and `dups`. I wrote the queries myself after reading the function list, so the absolute numbers are optimistic. With 26 queries one query is worth about 0.04.

Running `dups` on this repo found real copy-paste, like the `train` helpers in three tiny-transformer experiment scripts.

## Limitations

Python only. The call graph is name-based. It's lexical, so a query that shares no words with the code can miss, and neural embeddings are the obvious next experiment. `dups` is quadratic in the number of functions.
