"""Command-line interface for ragkit.

Commands:
  index <docs_dir> <index.json>   build an index from a directory
  ask <index.json> <query>        answer a question with citations
  eval <index.json> <golden.json> run the golden set and report metrics

Exit codes: 0 ok, 1 no results / eval below --fail-below threshold,
2 invalid index, query or arguments.
"""

from __future__ import annotations

import argparse
import json
import sys

from .answer import ExtractiveAnswerer, OpenAICompatibleAnswerer
from .eval import load_golden, report_text, run_eval
from .ingest import load_documents, load_index, save_index
from .retrieve import Retriever, select_for_answer, top_relative


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragkit", description="Local RAG over your own documents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_p = sub.add_parser("index", help="Build an index from a docs directory.")
    index_p.add_argument("docs_dir")
    index_p.add_argument("index_path")
    index_p.add_argument("--chunk-size", type=int, default=800)
    index_p.add_argument("--overlap", type=int, default=150)
    index_p.add_argument("--scoring", choices=["tfidf", "bm25"], default="bm25")

    ask_p = sub.add_parser("ask", help="Answer a question from an index.")
    ask_p.add_argument("index_path")
    ask_p.add_argument("query")
    ask_p.add_argument("--k", type=int, default=5)
    ask_p.add_argument("--scoring", choices=["tfidf", "bm25"], default="bm25")
    ask_p.add_argument(
        "--answerer", choices=["extractive", "llm"], default="extractive"
    )

    eval_p = sub.add_parser("eval", help="Evaluate against a golden set.")
    eval_p.add_argument("index_path")
    eval_p.add_argument("golden_path")
    eval_p.add_argument("--k", type=int, default=5)
    eval_p.add_argument("--scoring", choices=["tfidf", "bm25"], default="bm25")
    eval_p.add_argument("--fail-below", type=float, default=None,
                        help="Exit 1 if mean recall@k is below this value.")
    eval_p.add_argument("--json", dest="json_path", default=None,
                        help="Also write the full report as JSON to this path.")
    return parser


def _load_retriever(index_path: str, scoring: str) -> Retriever:
    try:
        chunks = load_index(index_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not chunks:
        print(f"error: index is empty: {index_path}", file=sys.stderr)
        raise SystemExit(2)
    return Retriever(chunks, scoring=scoring)


def cmd_index(args: argparse.Namespace) -> int:
    try:
        chunks = load_documents(args.docs_dir, args.chunk_size, args.overlap)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    save_index(
        chunks,
        args.index_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        scoring=args.scoring,
    )
    print(f"indexed {len(chunks)} chunks from {args.docs_dir} -> {args.index_path}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not args.query.strip():
        print("error: query must not be empty", file=sys.stderr)
        return 2
    retriever = _load_retriever(args.index_path, args.scoring)
    scored = retriever.search(args.query, k=args.k)
    if not scored:
        print("No relevant documents found.")
        return 1
    if args.answerer == "llm":
        try:
            answerer = OpenAICompatibleAnswerer()
        except RuntimeError as exc:  # pragma: no cover - needs a key
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        answerer = ExtractiveAnswerer()
    try:
        answer = answerer.answer(
            args.query, [s.chunk for s in select_for_answer(scored, args.query)]
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(answer.text)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    retriever = _load_retriever(args.index_path, args.scoring)
    try:
        golden = load_golden(args.golden_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not golden:
        print("error: golden set is empty", file=sys.stderr)
        return 2
    summary = run_eval(retriever, golden, k=args.k)
    print(report_text(summary, k=args.k))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        print(f"wrote {args.json_path}")
    if args.fail_below is not None and summary.mean_recall < args.fail_below:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "eval":
        return cmd_eval(args)
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
