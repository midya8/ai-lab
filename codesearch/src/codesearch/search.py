"""Query-time search: modes, filters, symbol boost, explanations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .chunker import Chunk
from .index import FIELDS, Index
from .tokens import looks_like_symbol, terms

MODES = ("name", "bm25", "bm25f", "lsa", "hybrid")
RRF_K = 60
TEST_PENALTY = 0.6
SYMBOL_BONUS = 1.0


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    matched: dict[str, list[str]] = field(default_factory=dict)  # field -> query terms found
    exact_symbol: bool = False

    def to_dict(self) -> dict:
        c = self.chunk
        return {"rank": self.rank, "id": c.id, "path": c.path, "qualname": c.qualname, "kind": c.kind,
                "start": c.start, "end": c.end, "score": round(float(self.score), 5),
                "signature": c.signature, "doc": c.doc.strip().split("\n")[0] if c.doc else "",
                "matched": self.matched, "exact_symbol": self.exact_symbol}


def query_symbols(query: str) -> list[str]:
    """Identifier-looking tokens the user typed (snake_case, camelCase, `x`, dotted.names, f())."""
    out = []
    for tok in re.split(r"\s+", query.strip()):
        if looks_like_symbol(tok):
            out.append(tok.strip("`()'\".,;:"))
    return out


class Searcher:
    def __init__(self, index: Index):
        self.index = index

    # ------------------------------------------------------------ scoring
    def _scores(self, mode: str, qterms: list[str]) -> np.ndarray:
        ix = self.index
        if mode == "name":
            return ix.name_only(qterms)
        if mode == "bm25":
            return ix.bm25_flat(qterms)
        if mode == "bm25f":
            return ix.bm25f(qterms)
        if mode == "lsa":
            return ix.lsa(qterms)
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")

    def _mask(self, kinds, path_prefix, include_tests, query_has_test) -> tuple[np.ndarray, np.ndarray]:
        """(allowed, multiplier). Tests are demoted, not hidden, unless the user filtered."""
        n = self.index.n
        allowed = np.ones(n, dtype=bool)
        mult = np.ones(n)
        for i, c in enumerate(self.index.chunks):
            if kinds and c.kind not in kinds:
                allowed[i] = False
            if path_prefix and not c.path.startswith(path_prefix):
                allowed[i] = False
            if c.is_test and not include_tests and not query_has_test:
                mult[i] = TEST_PENALTY
        return allowed, mult

    def search(self, query: str, k: int = 10, mode: str = "bm25f", kinds: list[str] | None = None,
               path_prefix: str | None = None, include_tests: bool = False,
               weights: tuple[float, float] = (1.0, 0.6)) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        ix = self.index
        qterms = terms(query)
        symbols = [s.lower() for s in query_symbols(query)]
        if not qterms and not symbols:
            return []
        allowed, mult = self._mask(kinds, path_prefix, include_tests, "test" in query.lower())

        if mode == "hybrid":
            fused = np.zeros(ix.n)
            for w, m in zip(weights, ("bm25f", "lsa")):
                s = self._scores(m, qterms) * mult
                s = np.where(allowed, s, 0.0)
                order = np.argsort(-s, kind="stable")[:100]
                for r, i in enumerate(order, start=1):
                    if s[i] > 0:
                        fused[i] += w / (RRF_K + r)
            scores = fused
        else:
            scores = self._scores(mode, qterms) * mult
        scores = np.where(allowed, scores, 0.0)

        exact = np.zeros(ix.n, dtype=bool)
        if symbols:
            for i, c in enumerate(ix.chunks):
                if allowed[i] and (c.name.lower() in symbols or c.qualname.lower() in symbols):
                    exact[i] = True
            scores = scores + SYMBOL_BONUS * exact * mult

        order = np.argsort(-scores, kind="stable")
        hits: list[Hit] = []
        qset = set(qterms)
        for i in order:
            if len(hits) >= k or scores[i] <= 0:
                break
            if not allowed[i]:
                continue
            matched = {f: sorted(qset & set(ix.fields[i][f])) for f in FIELDS}
            matched = {f: v for f, v in matched.items() if v}
            hits.append(Hit(ix.chunks[i], float(scores[i]), len(hits) + 1, matched, bool(exact[i])))
        return hits

    # ------------------------------------------------------ query by example
    def similar(self, ident: str, k: int = 10, include_tests: bool = True) -> list[Hit]:
        """Chunks most like the given one (by id, qualname, or unique name)."""
        idx = self.resolve(ident)
        ix = self.index
        qterms = ix.chunk_terms(idx)
        b = ix.bm25f(qterms)
        lsa = ix.lsa_neighbors(idx)
        fused = np.zeros(ix.n)
        for w, s in ((1.0, b), (1.0, lsa)):
            s = s.copy()
            s[idx] = 0
            for r, i in enumerate(np.argsort(-s, kind="stable")[:100], start=1):
                if s[i] > 0:
                    fused[i] += w / (RRF_K + r)
        fused[idx] = 0
        hits = []
        for i in np.argsort(-fused, kind="stable"):
            if len(hits) >= k or fused[i] <= 0:
                break
            if not include_tests and ix.chunks[i].is_test:
                continue
            hits.append(Hit(ix.chunks[i], float(fused[i]), len(hits) + 1))
        return hits

    # ------------------------------------------------------------ call graph
    def calls(self, ident: str, include_tests: bool = False) -> dict[str, list[Chunk]]:
        """Approximate call graph neighbours by identifier name.

        callees: functions/methods/classes whose bare name appears in this chunk's body.
        callers: chunks whose body mentions this chunk's bare name.
        Name-based, no type resolution: a name shared by several definitions links to all of
        them, and a mention that is not a call (a variable of the same name) counts.
        """
        ix = self.index
        idx = self.resolve(ident)
        me = ix.chunks[idx]
        by_name: dict[str, list[int]] = {}
        for i, c in enumerate(ix.chunks):
            if c.kind != "module":
                by_name.setdefault(c.name, []).append(i)
        used = set(me.ids)
        callees = [ix.chunks[j] for n in sorted(used) for j in by_name.get(n, [])
                   if j != idx and (include_tests or not ix.chunks[j].is_test)]
        callers = []
        if me.kind != "module":
            for i, c in enumerate(ix.chunks):
                if i == idx or c.kind == "module" or (c.is_test and not include_tests):
                    continue
                # a class chunk's ids include its methods' bodies; skip pure containers of me
                if me.name in c.ids and not (c.path == me.path and c.start <= me.start and c.end >= me.end):
                    callers.append(c)
        return {"callees": callees, "callers": callers}

    def undocumented(self, include_tests: bool = False, min_lines: int = 3) -> list[tuple[Chunk, int]]:
        """Public functions/methods/classes without a docstring, most-referenced first.

        Priority is the number of distinct chunks that mention the name (name-based, so
        approximate). Names starting with an underscore and bodies shorter than min_lines are skipped.
        """
        ix = self.index
        mentions: dict[str, set[int]] = {}
        for i, c in enumerate(ix.chunks):
            if c.kind == "module" or (c.is_test and not include_tests):
                continue
            for n in set(c.ids):
                mentions.setdefault(n, set()).add(i)
        out = []
        for i, c in enumerate(ix.chunks):
            if c.kind == "module" or c.doc or c.name.startswith("_"):
                continue
            if c.is_test and not include_tests:
                continue
            if c.end - c.start + 1 < min_lines:
                continue
            out.append((c, len(mentions.get(c.name, set()) - {i})))
        out.sort(key=lambda t: (-t[1], t[0].id))
        return out

    def unreferenced(self, include_tests: bool = False, ignore: tuple[str, ...] = ("main",)) -> list[Chunk]:
        """Dead-code candidates: functions/methods/classes whose name no other chunk mentions.

        Name-based, so it errs toward keeping code (a same-named variable elsewhere counts as a
        reference). Dunder methods, test functions, names in `ignore`, and chunks mentioned only
        by themselves or by a container that encloses them are reported as unreferenced only when
        no outside chunk mentions the name. Dynamic use (getattr, entry points, decorators that
        register) is invisible, so treat results as candidates to review.
        """
        ix = self.index
        out = []
        for i, c in enumerate(ix.chunks):
            if c.kind == "module" or c.is_test or c.name in ignore:
                continue
            if c.name.startswith("__") and c.name.endswith("__"):
                continue
            used = False
            for j, d in enumerate(ix.chunks):
                if j == i or d.kind == "module" or (d.is_test and not include_tests):
                    continue
                if c.name not in d.ids:
                    continue
                if d.path == c.path and d.start <= c.start and d.end >= c.end:
                    continue  # enclosing container
                used = True
                break
            if not used:
                out.append(c)
        out.sort(key=lambda c: c.id)
        return out

    def resolve(self, ident: str) -> int:
        ix = self.index
        if ident in ix.id_to_idx:
            return ix.id_to_idx[ident]
        cands = [i for i, c in enumerate(ix.chunks) if c.qualname == ident]
        if not cands:
            cands = [i for i, c in enumerate(ix.chunks) if c.name == ident]
        if not cands:
            raise KeyError(f"no chunk matches {ident!r}")
        if len(cands) > 1:
            names = ", ".join(ix.chunks[i].id for i in cands[:5])
            raise KeyError(f"{ident!r} is ambiguous; use path::qualname. Candidates: {names}")
        return cands[0]

    # ------------------------------------------------------------ duplicates
    def duplicates(self, threshold: float = 0.92, min_lines: int = 6, min_jaccard: float = 0.6,
                   include_tests: bool = False) -> list[tuple[Chunk, Chunk, float, float]]:
        """Near-duplicate function bodies: high LSA cosine AND high term-set overlap."""
        ix = self.index
        D = ix.lsa_matrix()
        if D is None:
            return []
        cand = [i for i, c in enumerate(ix.chunks)
                if c.kind in ("function", "method") and c.end - c.start + 1 >= min_lines
                and (include_tests or not c.is_test)]
        sets = {i: set(ix.fields[i]["body"]) | set(ix.fields[i]["sig"]) for i in cand}
        out = []
        arr = np.array(cand)
        if len(arr) < 2:
            return []
        sims = D[arr] @ D[arr].T
        for a in range(len(arr)):
            for b in range(a + 1, len(arr)):
                if sims[a, b] < threshold:
                    continue
                i, j = int(arr[a]), int(arr[b])
                if ix.chunks[i].path == ix.chunks[j].path and ix.chunks[i].qualname == ix.chunks[j].qualname:
                    continue
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j]) or 1
                jac = inter / union
                if jac >= min_jaccard:
                    out.append((ix.chunks[i], ix.chunks[j], float(sims[a, b]), jac))
        out.sort(key=lambda t: (-t[3], -t[2]))
        return out
