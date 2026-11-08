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
