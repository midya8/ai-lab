# codesearch

Semantic code search for Python repositories, from scratch (NumPy only). Ask in English or paste an identifier; get the *function* back, not a grep dump.

```bash
export PYTHONPATH=src
python -m codesearch search --root .. "write json to a temp file then rename it" --show
python -m codesearch search --root .. '`parse_diff`'                 # identifier → exact symbol first
python -m codesearch similar --root .. llm-router/llmrouter/breaker.py::CircuitBreaker
python -m codesearch dups --root ..                                  # near-duplicate functions
python -m codesearch calls --root .. Searcher.similar               # callers and callees (name-based)
python -m codesearch undocumented --root ..                    # public defs missing docstrings, most-referenced first
python -m codesearch eval bench/queries_dev.json --misses            # score the search itself
```

## How it works

1. **AST chunking.** Every function, method, class and module docstring is its own result with exact line range, signature, docstring, identifiers used, short string literals and the comments inside it. Decorators start the chunk; nested functions belong to their parent; files that do not parse are reported, not fatal.
2. **Identifier-aware terms.** `parseHTTPResponse`, `parse_http_response` and "parse an HTTP response" become the same terms (`pars http respons`) via camel/snake/acronym splitting and a small suffix stemmer.
3. **BM25F.** One score over six weighted fields: `name` (3.0), `owner` (class names of a method, 1.0), `doc` (2.0), `sig` (1.0), `body` (1.0), `path` (0.8), each with its own length normalisation. A function called `retry_with_backoff` beats one that merely mentions retries in a loop.
4. **LSA** (randomized SVD, no SciPy) and an RRF **hybrid** are implemented too; see the numbers below for why they are not the default.
5. **Identifier queries.** A backticked, `snake_case`, `camelCase` or `f()` token in the query that names a chunk exactly is pinned to the top and flagged `(exact symbol)`.
6. **Tests are demoted** (×0.6) unless you ask for them (`--tests`, or the word "test" in the query).
7. **Incremental index.** `.codesearch/index.json` stores chunks per file with a SHA-256; a refresh re-parses only changed files (tested with a spy parser), drops deleted ones, and is written atomically.
8. **Explanations.** Each hit says which query terms matched in which field; `--show` prints the code with matching lines marked `>`.
9. **Query by example** (`similar`): the chunk's own terms become the query, fused over BM25F and LSA neighbours. **`dups`** needs both LSA cosine ≥ 0.92 *and* term-set Jaccard ≥ 0.6, tests excluded by default.

## Measured, with the caveats

The corpus is this monorepo's other eight Python projects (1,357 chunks). Two query files in `bench/` — 26 natural-language queries each, labels are `path::qualname` (several where a behaviour legitimately lives in more than one place, e.g. both `CircuitBreaker` classes). `eval` refuses to run if a label no longer resolves.

| mode | dev R@1 | dev R@5 | dev MRR | held-out R@1 | held-out R@5 | held-out MRR |
|---|---|---|---|---|---|---|
| name-only (grep-like) | 0.192 | 0.308 | 0.247 | 0.077 | 0.154 | 0.122 |
| flat BM25 | 0.346 | 0.846 | 0.534 | 0.423 | 0.769 | 0.575 |
| **BM25F** (default) | **0.577** | **0.885** | **0.698** | 0.462 | 0.731 | 0.589 |
| LSA | 0.346 | 0.769 | 0.530 | 0.269 | 0.692 | 0.439 |
| hybrid (RRF 1 : 0.6) | 0.462 | 0.808 | 0.608 | 0.346 | 0.808 | 0.553 |

What to believe:

- **The queries were written by the author after reading the function list**, paraphrased on purpose but not blind. Treat absolute numbers as optimistic.
- **Dev vs held-out.** Field weights were set before looking at any results and a one-at-a-time sweep on dev (`experiments/sweeps.py`) moved MRR by at most ±0.02 except `doc ×2`, which hurt (0.698 → 0.612); I left the defaults alone. The held-out file was run once after choosing BM25F as the default; the first-run row was BM25F MRR 0.594 / hybrid 0.565 / LSA 0.472. I then added the `owner` field (so `CircuitBreaker.state` stops outscoring the class on its own name) — dev unchanged, held-out as in the table — so those numbers are no longer strictly held-out.
- **BM25F's lead over flat BM25 is clear on dev (MRR 0.70 vs 0.53) and almost gone on held-out (0.59 vs 0.58).** With 26 queries one query is ±0.04 R@1, so call that "BM25F is at least as good", not "better". Both are far above name-only.
- **LSA does not help at this scale, and hybrid is not better than BM25F alone.** Fusion weights, linear score mixing and a pseudo-relevance-feedback expansion (ad-hoc, not shipped) were all tried on dev; none beat plain BM25F. A 1.4k-chunk corpus of mostly short functions gives LSA too little to generalise from. LSA is still used for `similar` and `dups`, where the question is neighbourhood, not recall.
- Speed on this machine: sync of 123 files ≈ 0.5 s cold, index + LSA build ≈ 0.3 s, ≈ 0.3 ms per BM25F query.

Typical misses are vocabulary gaps ("how many tokens will this prompt use" → `estimate_tokens`, whose docstring says "~4 characters per token") and short generic names competing with test functions that repeat the query's words.

## Dogfooding `dups` on this repo

`python -m codesearch dups --root ..` surfaces genuine repetition: the `train` helpers copy-pasted across three `tiny-transformer/experiments/` scripts (Jaccard 0.86–0.96), `GPTLoRA.evaluate_loss` vs `AdapterHub.evaluate_loss`, and the near-identical constructors of `ResponseCache` and `SemanticCache`. Candidates, not verdicts: a human decides whether to merge them.

## Watch mode

`codesearch watch --root . --interval 1` keeps `.codesearch/index.json` fresh: it polls a cheap
(path, mtime, size) signature of the tree and runs the incremental sync only when it changes, so
unchanged files are never re-parsed. Polling is stdlib-only and portable; it is not instant
(latency up to one interval). `--cycles N` stops after N polls (used by the tests).

## JSON-lines server

`codesearch serve --root .` loads the index once and answers one JSON request per stdin line:

```bash
echo '{"id":1,"op":"search","query":"retry with backoff","k":3}' | python -m codesearch serve --root . --no-store
```

Ops: `search` (query, k, mode, kinds, path, tests), `similar` (ident, k), `stats`, `refresh` (re-sync
after files change), `ping`. Replies are `{"id", "ok", "result"}` or `{"id", "ok": false, "error"}`;
a malformed line gets an error reply and the server keeps going. The index is not auto-refreshed:
send `refresh` (or run `watch` alongside).

## Limitations

`calls` is a name-based call graph: no type or import resolution, so a name defined in several places links to all of them and a non-call mention (a variable with the same name) counts as a caller. Python only (JS/TS would need another parser). Lexical matching: a query that shares no words with the code or docstring and whose meaning LSA cannot bridge will miss. No embeddings from a neural model — that is the obvious next experiment and would be an optional extra, not a dependency. Dynamically generated functions are invisible. `dups` is quadratic in the number of functions (fine to ~10k).

## Layout

`src/codesearch/` — `tokens.py`, `chunker.py`, `store.py`, `index.py` (BM25F, flat BM25, randomized-SVD LSA), `search.py`, `evaluate.py`, `__main__.py`. `bench/` — query files and `build_queries.py` (authors labels, checks they resolve). `experiments/sweeps.py`. `tests/` — 62 tests (`CODESEARCH_MONOREPO=1` adds a check that the shipped labels still resolve in the full checkout).

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```
