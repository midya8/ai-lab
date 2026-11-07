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
