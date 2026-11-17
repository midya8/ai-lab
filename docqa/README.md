# docqa

Document Q&A that answers with sentences copied straight from your documents, each with a `[file:lines]` citation, and says "I don't know" when the documents don't cover the question. Built on [`ragkit`](../ragkit)'s BM25 retrieval. Standard library only.

```bash
cd docqa
python -m docqa ask ../ragkit/examples/docs "How long do I have to return a bike?"
python -m docqa eval ../ragkit/examples/docs eval/questions.json
python -m docqa serve ../ragkit/examples/docs --port 8000
python -m pytest -q
```

Each candidate sentence is scored by how much of the question's content it covers (idf-weighted). Below a threshold it abstains and exits 1. Questions like "how long" or "how many" prefer sentences that contain a number.

## How it does

These eval sets are small and I wrote them alongside the code, so they're smoke tests, not benchmarks:

- 6 basic questions: 3/3 answered, 3/3 correctly abstained.
- 13 broader questions: 10/10 answered, 3/3 abstained.
- 12 questions with typos: 7/12 without spelling correction, 12/12 with it. The corrector only uses words that appear in the documents, so it can't invent vocabulary, and every correction is reported.

Two things I tried that didn't matter: hybrid retrieval through [`vecdb`](../vecdb) gave identical results to BM25, since the answer stage is still lexical. Small synonym groups help with paraphrases, but generic words had to stay out because they produced confidently wrong answers.

## Other ways to use it

- As an [`agentkit`](../agentkit) tool (`ask_docs`) or an [`agentflow`](../agentflow) worker. In strict mode an abstention fails the task, so downstream tasks don't build on an unsupported answer.
- As an HTTP service with an answer cache and Prometheus metrics.
- With an optional LLM-written answer on top. A lexical check makes sure every number and most of each sentence's words appear in the evidence, and through [`llm-router`](../llm-router) it escalates to a stronger model only when the check fails. If nothing passes it falls back to the extractive answer.

Lexical matching is the main limit. It can't handle real paraphrase, and the grounding check catches invented figures but not a fluent sentence that recombines the evidence wrongly. The LLM tests use scripted providers only.
