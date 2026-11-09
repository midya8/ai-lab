# docqa

Grounded document Q&A with citations and an explicit "I don't know" mode.
Built on [ragkit](../ragkit) (BM25 retrieval); stdlib only.

- Answers are **verbatim sentences** from your documents, each with a `[file:lines]` citation. Nothing is generated.
- Each candidate sentence is scored by idf-weighted coverage of the question's content terms. If the best score is below `--min-coverage` (default 0.6), docqa **abstains** (exit code 1).
- "How long / how many / when" questions prefer sentences containing a number.

```bash
python -m docqa ask ../ragkit/examples/docs "How long do I have to return a bike?"
python -m docqa eval ../ragkit/examples/docs eval/questions.json
python -m pytest
```

Eval on the bundled 6-question set (3 answerable, 3 not): 3/3 answered correctly, 3/3 abstained. The set is tiny and tuned alongside the code, so treat it as a smoke test, not a benchmark. Lexical matching cannot handle paraphrase; see the dense-retrieval item in the roadmap.

## Paraphrase handling

Question terms match document terms through small synonym groups
(`SYNONYM_GROUPS` in `qa.py`), at 0.8 credit versus an exact match. Disable with
`DocQA(..., paraphrase=False)`. `eval/paraphrase.json` holds reworded questions;
abstention on unanswerable ones is still enforced. Generic words ("back",
"within") are deliberately not synonyms: they produced a confidently wrong answer.

## Broad benchmark (eval/broad.json)

13 questions across all six demo docs (10 answerable, 3 not). Current result:
answer rate 1.00 (10/10), abstain rate 1.00. The two earlier misses (yes/no phrasings)
were fixed by treating vague quantifiers ("certain", "amount") as filler and letting
"open" match "closed" (the sentence carries the polarity). Caveat: the set is small and
the fix was made while looking at it, so expect it not to generalise fully; unsupported
yes/no questions ("open on Mars?") still abstain (tested).

## Typo-tolerant questions (corpus-grounded spelling)

Lexical matching turns a typo into a missing term. Worse than abstaining, it can
produce a confidently wrong answer: on the bundled docs, "How long do I have to
*retrun* a bike?" without correction answers with the loaner-bike sentence.
`SpellCorrector` (`docqa/spell.py`) repairs the question before retrieval, using
only words that occur in the indexed documents (or their stems), so it cannot
invent vocabulary:

- words already in the corpus, stopwords, numbers and words under 4 letters are never touched;
- candidates are within optimal-string-alignment (Damerau-Levenshtein) distance 1
  (2 for words of 9+ letters); 4-letter words are fixed only for an adjacent swap ("opne" -> "open");
- ties break by corpus frequency, then alphabetically, so output is deterministic;
- every change is reported: `corrections` in the JSON result, and an
  `(interpreting as: retrun -> returns)` line in the CLI. `result.question` stays the caller's text.

```bash
python -m docqa ask ../ragkit/examples/docs "How long do I have to retrun a bike?"
python -m docqa eval ../ragkit/examples/docs eval/typos.json [--no-spell]
```

`eval/typos.json` (12 answerable questions with one typo each, plus 8 unanswerable
including typo'd and near-vocabulary ones): spelling off answers 7/12 correctly, spelling on 12/12;
abstention is 8/8 both ways, and the three clean benchmarks are unchanged. Caveat: the set was
written alongside the code (the "opne" rule came from a failure on it), so treat the lift as
a demonstration, not a generalisation estimate. Limits: it fixes single-word typos
against the corpus vocabulary, not word splits/joins ("shop hours" typed "shophours") or
a typo that lands on another valid corpus word. Disable with `DocQA(..., spell=False)` or `--no-spell`.

## Hybrid retrieval (`--retrieval hybrid`)

`DocQA(..., retrieval="hybrid")` swaps the retriever for BM25 fused (RRF) with a hashed
character-trigram dense index served by [vecdb](../vecdb). Result on the four eval sets
(spelling correction on or off): **identical** to BM25 everywhere (e.g. typos set with
spelling off: 7/12 both; with spelling on: 12/12 both). This is a null result: the answer
stage still scores sentences lexically, so better recall at the retrieval stage does not
rescue a misspelled term downstream, and this tiny corpus is already easy for BM25.
The option is kept because it is tested and cheap; the corpus is small, so do not read
this as evidence about larger corpora.

## Use as an agentkit tool

```python
from agentkit.tools import builtin_tools
from docqa.agent_tool import register

registry = register(builtin_tools(), "path/to/docs")   # adds `ask_docs`
registry.call("ask_docs", {"question": "How long do I have to return a bike?"})
```

The tool returns verbatim cited sentences or an explicit "I don't know", so an agent run can ground itself in documents and abstain when they are silent.

## Use as an agentflow worker

```python
from agentflow import Orchestrator, Task
from docqa.agentflow_worker import DocQAWorker

orch = Orchestrator([DocQAWorker("path/to/docs", strict=True)])
orch.run([Task("q", "docqa", "How long do I have to return a bike?")])
```

`DocQAWorker` has agentflow's `name` + `run(prompt)` shape, so it slots into a DAG and downstream tasks receive the cited answer via `{q}`. With `strict=True` an abstention raises `Abstained`: the task fails after its retries and dependents are skipped instead of building on an unsupported answer.

## HTTP service

```bash
python -m docqa serve path/to/docs --port 8000
curl -s localhost:8000/ask -d '{"question": "How long do I have to return a bike?"}'
curl -s localhost:8000/health
```

`POST /ask` returns the cited answer as JSON (an abstention is a normal 200 with `"answered": false`); malformed input gets 400, oversized bodies 413. Stdlib only, no extra dependencies.

### Answer cache and metrics

```bash
python -m docqa serve path/to/docs --cache-size 256 --cache-ttl 300   # --cache-size 0 disables
curl -s localhost:8000/stats      # JSON: requests, answered/abstained, cache, latency p50/p95/max
curl -s localhost:8000/metrics    # Prometheus text (counters, cache gauge, latency histogram)
```

Repeat questions are served from a thread-safe LRU + TTL cache and carry an `X-Cache: HIT|MISS` header. The key ignores case, punctuation and whitespace but keeps word order and every word, so reworded questions are cache misses by design (a wrong hit is worse than a miss). A hit echoes the caller's own phrasing in `question`. Latency percentiles are nearest-rank over the last 1024 `/ask` calls; unknown routes are bucketed as `other` so scanners cannot inflate label cardinality. Library use: `make_server(qa, cache=AnswerCache(...), metrics=Metrics())`.
