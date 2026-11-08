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

## Use as an agentkit tool

```python
from agentkit.tools import builtin_tools
from docqa.agent_tool import register

registry = register(builtin_tools(), "path/to/docs")   # adds `ask_docs`
registry.call("ask_docs", {"question": "How long do I have to return a bike?"})
```

The tool returns verbatim cited sentences or an explicit "I don't know", so an agent run can ground itself in documents and abstain when they are silent.
