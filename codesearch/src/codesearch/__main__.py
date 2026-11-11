"""codesearch command line: index, watch, search, similar, dups, eval, stats."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .evaluate import LabelError, check_labels, format_table, load_queries, misses, run_eval
from .index import Index
from .search import MODES, Hit, Searcher
from .store import sync

DEFAULT_STORE = ".codesearch/index.json"


def _load(args) -> tuple[Index, object]:
    root = Path(args.root).resolve()
    store = None if args.no_store else Path(args.store) if args.store else root / DEFAULT_STORE
    chunks, report = sync(root, store, include=args.include or None)
    if not chunks:
        raise SystemExit(f"no Python code found under {root}")
    return Index.build(chunks), report


def _snippet(root: Path, hit: Hit, terms_: set[str], max_lines: int = 12) -> list[str]:
    from .tokens import terms as tok

    c = hit.chunk
    try:
        lines = (root / c.path).read_text(errors="replace").splitlines()[c.start - 1:c.end]
    except OSError:
        return []
    out = []
    for n, line in enumerate(lines[:max_lines], start=c.start):
        mark = ">" if terms_ & set(tok(line)) else " "
        out.append(f"   {mark} {n:>4}  {line}")
    if len(lines) > max_lines:
        out.append(f"        ... {len(lines) - max_lines} more lines")
    return out


def _print_hits(hits: list[Hit], args, root: Path, qterms: set[str] | None = None) -> None:
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2))
        return
    if not hits:
        print("no matches")
        return
    for h in hits:
        c = h.chunk
        tag = " (exact symbol)" if h.exact_symbol else ""
        print(f"{h.rank:>2}. {c.path}:{c.start}-{c.end}  {c.qualname} [{c.kind}]  {h.score:.3f}{tag}")
        if c.signature and c.kind != "module":
            print(f"      {c.signature[:110]}")
        if c.doc:
            print(f"      {c.doc.strip().splitlines()[0][:110]}")
        if h.matched:
            print("      matched: " + "; ".join(f"{f}: {', '.join(v)}" for f, v in h.matched.items()))
        if getattr(args, "show", False) and qterms is not None:
            print("\n".join(_snippet(root, h, qterms)))


def cmd_index(args) -> int:
    root = Path(args.root).resolve()
    store = Path(args.store) if args.store else root / DEFAULT_STORE
    chunks, report = sync(root, store, include=args.include or None)
    print(f"{len(chunks)} chunks; {report.summary()}; store: {store}")
    for p, e in report.failed.items():
        print(f"  skipped {p}: {e}", file=sys.stderr)
    return 0


def cmd_watch(args) -> int:
    from .watch import watch

    root = Path(args.root).resolve()
    store = Path(args.store) if args.store else root / DEFAULT_STORE

    def report(rep, n):
        print(f"[{time.strftime('%H:%M:%S')}] {n} chunks; {rep.summary()}", flush=True)

    print(f"watching {root} every {args.interval}s (Ctrl-C to stop)")
    try:
        watch(root, store, include=args.include or None, interval=args.interval,
              on_sync=report, max_cycles=args.cycles)
    except KeyboardInterrupt:
        print("stopped")
    return 0


def cmd_search(args) -> int:
    from .tokens import terms

    ix, _ = _load(args)
    s = Searcher(ix)
    hits = s.search(" ".join(args.query), k=args.k, mode=args.mode,
                    kinds=args.kind or None, path_prefix=args.path, include_tests=args.tests)
    _print_hits(hits, args, Path(args.root).resolve(), set(terms(" ".join(args.query))))
    return 0 if hits else 1


def cmd_similar(args) -> int:
    ix, _ = _load(args)
    try:
        hits = Searcher(ix).similar(args.ident, k=args.k, include_tests=args.tests)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    _print_hits(hits, args, Path(args.root).resolve())
    return 0


def cmd_dups(args) -> int:
    ix, _ = _load(args)
    pairs = Searcher(ix).duplicates(threshold=args.threshold, min_lines=args.min_lines,
                                    min_jaccard=args.min_jaccard, include_tests=args.tests)
    if args.json:
        print(json.dumps([{"a": a.id, "b": b.id, "cosine": round(cos, 3), "jaccard": round(j, 3)}
                          for a, b, cos, j in pairs], indent=2))
        return 0
    for a, b, cos, j in pairs[:args.k]:
        print(f"jaccard {j:.2f} cosine {cos:.2f}\n  {a.path}:{a.start}  {a.qualname}\n  {b.path}:{b.start}  {b.qualname}")
    print(f"{len(pairs)} candidate pair(s)")
    return 0


def cmd_eval(args) -> int:
    meta, cases = load_queries(args.queries)
    root = Path(args.root or meta.get("root", ".")).resolve()
    if not args.root and not Path(meta.get("root", ".")).is_absolute():
        root = (Path(args.queries).resolve().parent / meta.get("root", ".")).resolve()
    chunks, _ = sync(root, None, include=args.include or meta.get("include"))
    ix = Index.build(chunks)
    try:
        check_labels(ix, cases)
    except LabelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    modes = tuple(args.modes.split(",")) if args.modes else MODES
    scores = run_eval(ix, cases, modes)
    print(f"{len(chunks)} chunks, {len(cases)} queries\n")
    print(format_table(scores))
    if args.misses:
        best = next((s for s in scores if s.mode == "bm25f"), scores[0])
        print(f"\nQueries whose first relevant hit is outside the top 3 ({best.mode}):")
        print("\n".join(misses(ix, cases, best, 0)) or "  none")
    return 0


def cmd_stats(args) -> int:
    ix, report = _load(args)
    kinds: dict[str, int] = {}
    for c in ix.chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    files = len({c.path for c in ix.chunks})
    print(f"{len(ix.chunks)} chunks in {files} files: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"vocabulary: {len(ix.postings)} terms; documented: {sum(1 for c in ix.chunks if c.doc)}/{len(ix.chunks)}")
    print("sync: " + report.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codesearch", description="Semantic code search for Python repositories.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, root=True):
        sp.add_argument("--root", default=".", help="repository root (default .)")
        sp.add_argument("--store", help=f"index file (default <root>/{DEFAULT_STORE})")
        sp.add_argument("--no-store", action="store_true", help="do not read or write an index file")
        sp.add_argument("--include", action="append", help="only index these path prefixes (repeatable)")
        sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("index", help="build or refresh the on-disk index")
    common(sp); sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("watch", help="keep the on-disk index fresh while files change")
    common(sp)
    sp.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    sp.add_argument("--cycles", type=int, help="stop after this many polls (default: run until Ctrl-C)")
    sp.set_defaults(fn=cmd_watch)

    sp = sub.add_parser("search", help="natural-language or identifier search")
    sp.add_argument("query", nargs="+")
    common(sp)
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--mode", choices=MODES, default="bm25f")
    sp.add_argument("--kind", action="append", choices=["function", "method", "class", "module"])
    sp.add_argument("--path", help="only paths starting with this prefix")
    sp.add_argument("--tests", action="store_true", help="do not demote test code")
    sp.add_argument("--show", action="store_true", help="print the code, marking lines with query terms")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("similar", help="functions most like a given one (query by example)")
    sp.add_argument("ident", help="path::qualname, qualname, or a unique name")
    common(sp)
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--no-tests", dest="tests", action="store_false", help="leave test code out of the results")
    sp.set_defaults(fn=cmd_similar)

    sp = sub.add_parser("dups", help="near-duplicate functions")
    common(sp)
    sp.add_argument("-k", type=int, default=20)
    sp.add_argument("--threshold", type=float, default=0.92, help="minimum LSA cosine")
    sp.add_argument("--min-jaccard", type=float, default=0.6)
    sp.add_argument("--min-lines", type=int, default=6)
    sp.add_argument("--tests", action="store_true", help="include test code (excluded by default)")
    sp.set_defaults(fn=cmd_dups)

    sp = sub.add_parser("eval", help="score search quality on a labelled query file")
    sp.add_argument("queries")
    sp.add_argument("--root")
    sp.add_argument("--include", action="append")
    sp.add_argument("--modes", help="comma list, default all: " + ",".join(MODES))
    sp.add_argument("--misses", action="store_true")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("stats", help="index statistics")
    common(sp); sp.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
