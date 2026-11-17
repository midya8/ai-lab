# rag-evals

The retrieval half of a RAG system, plus a way to measure it. BM25, an LSA dense retriever (TF-IDF + truncated SVD in NumPy), and a hybrid of the two with reciprocal rank fusion. Scored with recall@k, MRR and nDCG@k on labelled questions.

```bash
cd rag-evals
pip install -r requirements.txt
PYTHONPATH=src python -m ragevals.cli --k 3 --method all
PYTHONPATH=src python -m ragevals.cli --query "what is QLoRA"
PYTHONPATH=src python -m ragevals.cli --docs data/hard_docs --dataset data/hard_eval.json --method all --k 3
python -m pytest -q
```

The bundled starter set is too easy (recall@3 = 1.0), so I wrote a harder one: 8 topically similar docs and 19 paraphrased questions with deliberate vocabulary mismatch. On that, BM25 and the hybrid both get recall@3 0.842 / MRR 0.866, and LSA gets 0.842 / 0.839. So the dense side didn't add anything here.

A chunk-size sweep (`--sweep 15,25,40,60,100`) shows MRR peaking around 40 words and dropping to 0.784 at 15, where chunks lose context. There's also a lexical faithfulness check that splits an answer into claims and looks for support in the retrieved context (`--faithfulness data/answers.json`).

It's a tiny hand-written corpus, so one question is worth ~0.05 and this works better as a regression canary than a leaderboard. [`ragkit`](../ragkit) builds answering on top of the same idea.
