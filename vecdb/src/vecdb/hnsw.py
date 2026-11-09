"""Hierarchical Navigable Small World graph (Malkov & Yashunin, 2018), from the paper."""
from __future__ import annotations

import heapq
import math

import numpy as np

from .util import Metric, as_matrix


class HNSWIndex:
    """Multi-layer proximity graph.

    Each vector draws a top layer ``floor(-ln(U) / ln(M))``. Insertion greedily descends from the
    entry point, then at every layer up to the vector's own searches ``ef_construction`` candidates
    and links the new node to ``M`` of them chosen by the *diversity heuristic* (Algorithm 4: keep
    a candidate only if it is closer to the new node than to every neighbour already kept), which
    keeps links pointing in different directions instead of all into the same tight cluster.
    Layer 0 allows ``2M`` links per node. Deletion is a tombstone: a deleted node still routes
    traffic but is never returned.

    Filtered search keeps traversing through non-matching nodes but only collects matching ones;
    when the filter is very selective (fewer than ``brute_force_below`` of the live rows; default 0.15, the point where the two cost the same number of distance evaluations in experiments/results.md) it
    abandons the graph and scans the matching rows exactly, because a graph walk would have to
    cross most of the index to find ``ef`` matches.
    """

    kind = "hnsw"

    def __init__(self, dim: int, metric: str = "l2", M: int = 16, ef_construction: int = 100,
                 seed: int = 0, brute_force_below: float = 0.15):
        if M < 2:
            raise ValueError("M must be >= 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        self.dim = dim
        self.metric = Metric(metric)
        self.M, self.M0 = M, 2 * M
        self.ef_construction = ef_construction
        self.seed = seed
        self.brute_force_below = brute_force_below
        self._mL = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)
        self._vecs = np.empty((16, dim), dtype=np.float32)
        self.ntotal = 0
        self.deleted = np.zeros(16, dtype=bool)
        self.levels: list[int] = []
        self.links: list[dict[int, list[int]]] = []  # links[level][node] -> neighbour list
        self.entry = -1
        self.max_level = -1
        self.last_stats: dict = {}

    # -- bookkeeping ---------------------------------------------------
    @property
    def n_live(self) -> int:
        return int((~self.deleted[:self.ntotal]).sum())

    @property
    def vecs(self) -> np.ndarray:
        return self._vecs[:self.ntotal]

    def _grow(self, need: int) -> None:
        cap = self._vecs.shape[0]
        if need <= cap:
            return
        new_cap = max(need, cap * 2)
        v = np.empty((new_cap, self.dim), dtype=np.float32)
        v[:self.ntotal] = self._vecs[:self.ntotal]
        d = np.zeros(new_cap, dtype=bool)
        d[:self.ntotal] = self.deleted[:self.ntotal]
        self._vecs, self.deleted = v, d

    def _dist_to(self, q: np.ndarray, ids) -> np.ndarray:
        V = self._vecs[ids]
        if self.metric.name == "ip":
            return -(V @ q)
        diff = V - q
        return np.einsum("ij,ij->i", diff, diff)

    # -- construction --------------------------------------------------
    def add(self, X) -> np.ndarray:
        A = self.metric.prep(as_matrix(X, self.dim))
        start = self.ntotal
        self._grow(start + len(A))
        for a in A:
            node = self.ntotal
            self._vecs[node] = a
            self.deleted[node] = False
            self.ntotal += 1
            level = int(-math.log(1.0 - self._rng.random()) * self._mL)
            self.levels.append(level)
            while len(self.links) <= level:
                self.links.append({})
            for l in range(level + 1):
                self.links[l][node] = []
            self._insert(node, level)
        return np.arange(start, self.ntotal, dtype=np.int64)

    def _insert(self, node: int, level: int) -> None:
        if self.entry < 0:
            self.entry, self.max_level = node, level
            return
        q = self._vecs[node]
        ep = [self.entry]
        for lev in range(self.max_level, level, -1):
            ep = [self._search_layer(q, ep, 1, lev)[0][1]]
        for lev in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, ep, self.ef_construction, lev, exclude=node)
            cap = self.M0 if lev == 0 else self.M
            chosen = self._select(found, self.M)
            self.links[lev][node] = list(chosen)
            for nb in chosen:
                lst = self.links[lev][nb]
                lst.append(node)
                if len(lst) > cap:
                    d = self._dist_to(self._vecs[nb], lst)
                    cands = sorted(zip(d.tolist(), lst))
                    self.links[lev][nb] = self._select(cands, cap)
            ep = [n for _, n in found]
        if level > self.max_level:
            self.entry, self.max_level = node, level

    def _select(self, cands, m: int) -> list[int]:
        """Diversity heuristic over (distance-to-query, id) pairs sorted ascending."""
        if len(cands) <= m:
            return [c for _, c in cands]
        ids = np.array([c for _, c in cands])
        dq = np.array([d for d, _ in cands])
        V = self._vecs[ids]
        P = self.metric.dist(V, V)
        kept: list[int] = []
        skipped: list[int] = []
        for i in range(len(ids)):
            if len(kept) >= m:
                break
            if all(P[i, j] > dq[i] for j in kept):
                kept.append(i)
            else:
                skipped.append(i)
        for i in skipped:  # keepPrunedConnections: top up with the closest rejected
            if len(kept) >= m:
                break
            kept.append(i)
        return [int(ids[i]) for i in kept]

    def _search_layer(self, q, eps, ef, level, allow=None, exclude=-1):
        """Best-first search on one layer. Returns [(dist, node)] ascending, at most ef long.

        With ``allow`` (bool array over rows) only allowed nodes enter the result set, but
        every node is still traversed.
        """
        visited = set(eps)
        visited.add(exclude)
        eps = [e for e in eps if e != exclude]
        d0 = self._dist_to(q, np.array(eps, dtype=np.int64)).tolist()
        cand = list(zip(d0, eps))
        heapq.heapify(cand)
        res = [(-d, e) for d, e in cand if allow is None or allow[e]]
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        links = self.links[level]
        evals = len(eps)
        while cand:
            d, c = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break
            nbrs = [n for n in links.get(c, ()) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            ds = self._dist_to(q, np.array(nbrs, dtype=np.int64)).tolist()
            evals += len(nbrs)
            for n, dn in zip(nbrs, ds):
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, n))
                    if allow is None or allow[n]:
                        heapq.heappush(res, (-dn, n))
                        if len(res) > ef:
                            heapq.heappop(res)
        self._evals += evals
        return sorted((-nd, n) for nd, n in res)

    _evals = 0

    # -- deletion -------------------------------------------------------
    def remove(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.ntotal):
            raise IndexError("row out of range")
        self.deleted[rows] = True

    def repair(self) -> dict:
        """Heal the graph around tombstones (the deleted nodes keep routing until this runs).

        For every live node with a deleted neighbour, the deleted neighbours are replaced by *their*
        live neighbours as candidates, and the node's list is re-chosen from (surviving links +
        candidates) with the diversity heuristic. Deleted nodes are then unlinked, and the entry
        point moves to the highest live node. Rows keep their ids and the deleted vectors stay in
        memory (ids are positional); only graph edges are rewritten.
        """
        dead = self.deleted[:self.ntotal]
        fixed = 0
        for lev, lv in enumerate(self.links):
            cap = self.M0 if lev == 0 else self.M
            for node in list(lv):
                if dead[node]:
                    continue
                nbrs = lv[node]
                if not any(dead[x] for x in nbrs):
                    continue
                pool = {x for x in nbrs if not dead[x]}
                for x in nbrs:
                    if dead[x]:
                        pool.update(y for y in lv.get(x, ()) if not dead[y] and y != node)
                ids = sorted(pool)
                d = self._dist_to(self._vecs[node], ids)
                cands = sorted(zip(d.tolist(), ids))
                lv[node] = self._select(cands, cap) if len(ids) > cap else [n for _, n in cands]
                fixed += 1
            for node in [n for n in lv if dead[n]]:
                lv[node] = []
        live_levels = [(self.levels[n], n) for n in range(self.ntotal) if not dead[n]]
        if live_levels:
            top, ent = max(live_levels)
            self.entry, self.max_level = ent, top
        else:
            self.entry, self.max_level = -1, -1
        return {"nodes_repaired": fixed, "entry": self.entry}

    def reconstruct(self, rows) -> np.ndarray:
        return self._vecs[np.asarray(rows, dtype=np.int64)].copy()

    # -- search ---------------------------------------------------------
    def search(self, Q, k: int = 10, ef: int = 64, mask=None):
        if k < 1:
            raise ValueError("k must be >= 1")
        Qm = self.metric.prep(as_matrix(Q, self.dim))
        out_i = np.full((len(Qm), k), -1, dtype=np.int64)
        out_d = np.full((len(Qm), k), np.inf, dtype=np.float32)
        if self.entry < 0:
            return out_i, out_d
        live = ~self.deleted[:self.ntotal]
        allow = live
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.ntotal,):
                raise ValueError(f"mask must have shape ({self.ntotal},), got {m.shape}")
            allow = live & m
        n_allowed = int(allow.sum())
        use_allow = None if n_allowed == self.ntotal else allow
        brute = use_allow is not None and n_allowed < self.brute_force_below * max(self.n_live, 1)
        ef = max(int(ef), k)
        self._evals = 0
        if brute:
            rows = np.flatnonzero(allow)
            self._evals = len(rows) * len(Qm)
            for qi, q in enumerate(Qm):
                if len(rows) == 0:
                    break
                d = self._dist_to(q, rows)
                o = np.lexsort((rows, d))[:k]
                out_i[qi, :len(o)] = rows[o]
                out_d[qi, :len(o)] = d[o]
        else:
            for qi, q in enumerate(Qm):
                ep = [self.entry]
                for lev in range(self.max_level, 0, -1):
                    ep = [self._search_layer(q, ep, 1, lev)[0][1]]
                res = self._search_layer(q, ep, ef, 0, allow=use_allow)[:k]
                for j, (d, n) in enumerate(res):
                    out_i[qi, j], out_d[qi, j] = n, d
        self.last_stats = {"scanned": self._evals / max(len(Qm), 1), "ef": ef, "brute_force": bool(brute)}
        return out_i, self.metric.report(out_d)

    # -- diagnostics ----------------------------------------------------
    def degree_stats(self) -> dict:
        l0 = [len(v) for v in self.links[0].values()] if self.links else [0]
        return {"mean_degree_l0": float(np.mean(l0)), "max_degree_l0": int(max(l0)),
                "levels": self.max_level + 1, "nodes_above_l0": int(sum(1 for l in self.levels if l > 0))}

    def memory_bytes(self) -> int:
        n_links = sum(len(v) for lv in self.links for v in lv.values())
        return int(self.vecs.nbytes + 4 * n_links)

    # -- persistence ----------------------------------------------------
    def _state(self):
        arrays = {"vecs": self.vecs, "deleted": self.deleted[:self.ntotal],
                  "levels": np.array(self.levels, dtype=np.int32)}
        for l, lv in enumerate(self.links):
            nodes = np.array(sorted(lv), dtype=np.int64)
            lens = np.array([len(lv[n]) for n in nodes], dtype=np.int64)
            flat = np.array([x for n in nodes for x in lv[n]], dtype=np.int64)
            arrays[f"l{l}_nodes"], arrays[f"l{l}_lens"], arrays[f"l{l}_flat"] = nodes, lens, flat
        meta = {"dim": self.dim, "metric": self.metric.name, "M": self.M, "ef_construction": self.ef_construction,
                "seed": self.seed, "brute_force_below": self.brute_force_below, "entry": self.entry,
                "max_level": self.max_level, "n_levels": len(self.links), "ntotal": self.ntotal}
        return arrays, meta

    @classmethod
    def _from_state(cls, arrays, meta):
        idx = cls(meta["dim"], meta["metric"], meta["M"], meta["ef_construction"], meta["seed"],
                  meta["brute_force_below"])
        n = meta["ntotal"]
        idx._grow(max(n, 16))
        idx._vecs[:n] = arrays["vecs"]
        idx.deleted[:n] = arrays["deleted"]
        idx.ntotal = n
        idx.levels = [int(x) for x in arrays["levels"]]
        idx.entry, idx.max_level = meta["entry"], meta["max_level"]
        for l in range(meta["n_levels"]):
            nodes, lens, flat = arrays[f"l{l}_nodes"], arrays[f"l{l}_lens"], arrays[f"l{l}_flat"]
            off = np.concatenate([[0], np.cumsum(lens)])
            idx.links.append({int(nd): [int(x) for x in flat[off[i]:off[i + 1]]] for i, nd in enumerate(nodes)})
        idx._rng = np.random.default_rng([meta["seed"], n])
        return idx
