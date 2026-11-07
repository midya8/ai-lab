"""python -m ragevals.cli --docs data/docs --dataset data/eval.json --k 3"""
from __future__ import annotations
import argparse, json, pathlib

from .chunking import chunk_text
from .metrics import evaluate
from .retrieval import BM25Index


def build_index(docs_dir: str, max_words: int = 40) -> BM25Index:
    chunks = []
    for p in sorted(pathlib.Path(docs_dir).glob("*.txt")):
        chunks += chunk_text(p.stem, p.read_text(), max_words=max_words)
    return BM25Index(chunks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate BM25 retrieval over a docs folder")
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--dataset", default="data/eval.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--query", help="run a single query instead of the eval")
    a = ap.parse_args(argv)
    idx = build_index(a.docs)
    if a.query:
        for c, s in idx.search(a.query, a.k):
            print(f"{s:6.2f}  {c.id}  {c.text[:90]}")
        return 0
    data = json.loads(pathlib.Path(a.dataset).read_text())
    res = evaluate(lambda q, k: [c.doc_id for c, _ in idx.search(q, k)], data, a.k)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
