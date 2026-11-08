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
