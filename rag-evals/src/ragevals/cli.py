"""python -m ragevals.cli --docs data/docs --dataset data/eval.json --k 3"""
from __future__ import annotations
import argparse, json, pathlib

from .chunking import chunk_text
from .metrics import evaluate
from .retrieval import BM25Index
from .dense import LSAIndex, HybridIndex


METHODS = {"bm25": BM25Index, "lsa": LSAIndex, "hybrid": HybridIndex}


def load_chunks(docs_dir: str, max_words: int = 40):
    chunks = []
    for p in sorted(pathlib.Path(docs_dir).glob("*.txt")):
        chunks += chunk_text(p.stem, p.read_text(), max_words=max_words)
    return chunks


def build_index(docs_dir: str, max_words: int = 40, method: str = "bm25"):
    return METHODS[method](load_chunks(docs_dir, max_words))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate BM25 retrieval over a docs folder")
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--dataset", default="data/eval.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--method", choices=[*METHODS, "all"], default="bm25")
    ap.add_argument("--query", help="run a single query instead of the eval")
    ap.add_argument("--faithfulness", metavar="ANSWERS_JSON", help="score answer faithfulness/relevance from a JSON file")
    ap.add_argument("--sweep", metavar="SIZES", help="comma-separated chunk sizes (words) to compare, e.g. 15,40,100")
    a = ap.parse_args(argv)
    if a.sweep:
        from .sweep import sweep, to_markdown
        sizes = [int(x) for x in a.sweep.split(",")]
        data = json.loads(pathlib.Path(a.dataset).read_text())
        methods = list(METHODS) if a.method == "all" else [a.method]
        print(to_markdown(sweep(a.docs, data, sizes, methods, a.k), a.k))
        return 0
    if a.faithfulness:
        from .faithfulness import evaluate_answers
        print(json.dumps(evaluate_answers(json.loads(pathlib.Path(a.faithfulness).read_text())), indent=2))
        return 0
    if a.method == "all":
        data = json.loads(pathlib.Path(a.dataset).read_text())
        table = {}
        for m in METHODS:
            idx = build_index(a.docs, method=m)
            table[m] = evaluate(lambda q, k, idx=idx: [c.doc_id for c, _ in idx.search(q, k)], data, a.k)
        print(json.dumps(table, indent=2))
        return 0
    idx = build_index(a.docs, method=a.method)
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
