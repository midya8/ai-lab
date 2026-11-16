"""A paged KV cache (the vLLM idea) with block tables, prefix sharing and copy-on-write.

Keys/values live in a fixed pool of ``block_size``-token blocks. Each sequence owns a *block
table* - a list of physical block ids - so sequences can grow without a contiguous reservation,
freed blocks are reusable immediately, and forked sequences (beam search, shared system prompt)
share their prefix blocks by reference count. Attention runs directly over the blocks: each block
is one key tile of the online softmax, merged exactly with :func:`merge_partials`.
"""
from __future__ import annotations

import numpy as np

from .flash import attend_partial, finalize, merge_partials


class OutOfBlocks(RuntimeError):
    pass


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int, n_kv_heads: int, head_dim: int,
                 dtype=np.float64):
        self.block_size, self.n_kv_heads, self.head_dim = block_size, n_kv_heads, head_dim
        self.k = np.zeros((num_blocks, block_size, n_kv_heads, head_dim), dtype=dtype)
        self.v = np.zeros_like(self.k)
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.refcount = np.zeros(num_blocks, dtype=int)
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        self._next_id = 0
        self.copies = 0       # copy-on-write events, for tests and accounting

    # -- allocation ---------------------------------------------------------------------------
    def _alloc(self) -> int:
        if not self.free_blocks:
            raise OutOfBlocks("KV cache pool exhausted")
        b = self.free_blocks.pop()
        self.refcount[b] = 1
        return b

    def _release(self, b: int) -> None:
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free_blocks.append(b)

    def new_sequence(self) -> int:
        sid = self._next_id
        self._next_id += 1
        self.tables[sid], self.lengths[sid] = [], 0
        return sid

    def free(self, sid: int) -> None:
        for b in self.tables.pop(sid):
            self._release(b)
        del self.lengths[sid]

    def fork(self, sid: int) -> int:
        """New sequence sharing every block of ``sid`` (no data copied)."""
        child = self.new_sequence()
        self.tables[child] = list(self.tables[sid])
        self.lengths[child] = self.lengths[sid]
        for b in self.tables[child]:
            self.refcount[b] += 1
        return child

    # -- writes -------------------------------------------------------------------------------
    def append(self, sid: int, k, v) -> None:
        """Append ``T`` tokens; k, v have shape (T, n_kv_heads, head_dim)."""
        k, v = np.asarray(k), np.asarray(v)
        needed_new = -(-(self.lengths[sid] + len(k)) // self.block_size) - len(self.tables[sid])
        shared_tail = (self.lengths[sid] % self.block_size != 0
                       and self.refcount[self.tables[sid][-1]] > 1)
        if needed_new + int(shared_tail) > len(self.free_blocks):
            raise OutOfBlocks("KV cache pool exhausted")
        for t in range(len(k)):
            n = self.lengths[sid]
            slot = n % self.block_size
            if slot == 0:
                self.tables[sid].append(self._alloc())
            else:
                tail = self.tables[sid][-1]
                if self.refcount[tail] > 1:                      # copy-on-write
                    fresh = self._alloc()
                    self.k[fresh], self.v[fresh] = self.k[tail], self.v[tail]
                    self._release(tail)
                    self.tables[sid][-1] = fresh
                    self.copies += 1
            b = self.tables[sid][-1]
            self.k[b, slot], self.v[b, slot] = k[t], v[t]
            self.lengths[sid] = n + 1

    # -- reads --------------------------------------------------------------------------------
    def gather(self, sid: int):
        """Contiguous (K, V), each (n_kv_heads, length, head_dim) - for tests and debugging."""
        n = self.lengths[sid]
        if n == 0:
            e = np.zeros((self.n_kv_heads, 0, self.head_dim), dtype=self.k.dtype)
            return e, e.copy()
        ks = np.concatenate([self.k[b] for b in self.tables[sid]])[:n]
        vs = np.concatenate([self.v[b] for b in self.tables[sid]])[:n]
        return ks.transpose(1, 0, 2), vs.transpose(1, 0, 2)

    def attend(self, sid: int, q, *, causal=True, window=None, alibi_slopes=None, scale=None):
        """Attention of ``q`` (H, nq, d) - the *last* nq tokens of the sequence - over the paged
        cache, one physical block per key tile, never gathering the cache. Returns (out, lse)."""
        q = np.asarray(q)
        n, bs = self.lengths[sid], self.block_size
        off = n - q.shape[1]
        parts = []
        for bi, b in enumerate(self.tables[sid]):
            valid = min(bs, n - bi * bs)
            kb = self.k[b, :valid].transpose(1, 0, 2)
            vb = self.v[b, :valid].transpose(1, 0, 2)
            parts.append(attend_partial(q, kb, vb, causal=causal, window=window,
                                        alibi_slopes=alibi_slopes, q_offset=off,
                                        k_start=bi * bs, block_q=max(q.shape[1], 1),
                                        block_k=bs, scale=scale, dtype=self.k.dtype.type))
        if not parts:
            raise ValueError("empty sequence")
        return finalize(*merge_partials(parts))

    # -- accounting ---------------------------------------------------------------------------
    def blocks_in_use(self) -> int:
        return int((self.refcount > 0).sum())

    def utilisation(self) -> dict:
        """Token slots actually holding data vs allocated, counting a shared block once."""
        used = self.blocks_in_use()
        return {"blocks_in_use": used, "logical_tokens": sum(self.lengths.values()),
                "physical_slots": used * self.block_size,
                "internal_waste_slots": used * self.block_size - self._physical_tokens()}

    def _physical_tokens(self) -> int:
        """Distinct (block, slot) pairs holding live data."""
        seen = {}
        for s, tab in self.tables.items():
            for i, b in enumerate(tab):
                seen[b] = max(seen.get(b, 0), min(self.block_size, self.lengths[s] - i * self.block_size))
        return sum(seen.values())
