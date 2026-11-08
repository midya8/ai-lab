# rag-evals

Dependency-free retrieval pipeline for RAG with a built-in evaluation harness.

- **Chunker**: sentence-aware sliding window with overlap
- **Retriever**: Okapi BM25 implemented from scratch
- **Dense retriever**: TF-IDF + truncated SVD (LSA) in NumPy
- **Hybrid**: BM25 + LSA fused with Reciprocal Rank Fusion
- **Metrics**: Recall@k, MRR, nDCG@k over a labeled question set
- **CLI**: run the eval or ad-hoc queries

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m ragevals.cli --k 3            # eval on data/eval.json
PYTHONPATH=src python -m ragevals.cli --query "what is QLoRA"
PYTHONPATH=src python -m ragevals.cli --method all    # compare bm25 / lsa / hybrid
python -m pytest
```

Sample result on the bundled corpus: recall@3 = 1.0, MRR = 1.0 (7 questions, 5 docs).
Add your own `.txt` files to `data/docs` and questions to `data/eval.json`.

## Answer faithfulness (v0.3)
`ragevals.faithfulness` splits an answer into claims and checks each against retrieved context
(lexical support, threshold configurable; pass `supports=` to plug in an NLI/LLM judge). Also reports answer relevance.
```bash
PYTHONPATH=src python -m ragevals.cli --faithfulness data/answers.json
```

## Harder benchmark (v0.4)
`data/hard_docs` (8 topically adjacent docs) + `data/hard_eval.json` (19 paraphrased questions with deliberate vocabulary mismatch) so the original 1.0 scores are no longer the ceiling.
```bash
PYTHONPATH=src python -m ragevals.cli --docs data/hard_docs --dataset data/hard_eval.json --method all --k 3
```
Result (recall@3 / MRR): BM25 0.842 / 0.866, LSA 0.842 / 0.839, hybrid 0.842 / 0.866. Small hand-written corpus: treat as a regression canary, not a leaderboard. A test asserts the set stays unsaturated.

## Chunk-size sweep (v0.5)
`--sweep` re-indexes the corpus at several chunk sizes and prints a Markdown table (`ragevals.sweep.sweep/best` for programmatic use).
```bash
PYTHONPATH=src python -m ragevals.cli --docs data/hard_docs --dataset data/hard_eval.json --method all --sweep 15,25,40,60,100
```
On the hard set: MRR peaks at 40 words for BM25/hybrid (0.866) and drops to 0.784 at 15 words, where chunks lose context; recall@3 is flat-ish (0.84-0.89) because docs are short. Tiny corpus, so differences of one question (~0.05) are noise.
